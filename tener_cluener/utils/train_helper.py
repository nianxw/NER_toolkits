import os
import json
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import (DataLoader,
                              RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

from utils.data_helper import NER_dataset, load_and_cache_examples, collate_fn
from utils.common import logger, seed_everything, json_to_text
from utils.ner_metrics import SeqEntityScore
from utils.ner_metrics import get_entity_bios


def train(args, processor, model):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = NER_dataset(load_and_cache_examples(args, processor, data_type='train'),
                                args.train_max_seq_len)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    t_total = len(train_dataloader) * args.epoch

    transformer_param_optimizer = list(model.transformer.parameters())
    crf_param_optimizer = list(model.crf.parameters())
    linear_param_optimizer = list(model.out_fc.parameters())

    optimizer_grouped_parameters = [
        {'params': transformer_param_optimizer,  'lr': args.learning_rate},
        {'params': crf_param_optimizer,  'lr': args.crf_learning_rate},
        {'params': linear_param_optimizer,  'lr': args.crf_learning_rate},
    ]
    args.warmup_steps = int(t_total * args.warmup_rate)
    if args.optim == 'sgd:':
        optimizer = optim.SGD(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              momentum=args.momentum_rate)
    elif args.optim == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed) = %d",
                args.train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0

    best_f1 = 0.0
    tr_loss = 0.0
    model.zero_grad()
    # Added here for reproductibility (even between python 2 and 3)
    seed_everything(args.seed)
    for index in range(int(args.epoch)):
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "input_mask": batch[1],
                      "labels": batch[2], 'input_lens': batch[3]}
            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            model.zero_grad()
            if global_step % args.log_steps == 0:
                logger.info("training porcess —— epoch:%d —— global_step-%d —— loss-%.4f" % (index+1, global_step+1, loss.item()))
            global_step += 1
        if args.local_rank in [-1, 0]:
            # Log metrics
            print(" ")
            if args.local_rank == -1:
                # Only evaluate when single GPU otherwise metrics may not average well
                eval_results = evaluate(args, processor, model)
                if eval_results['f1'] > best_f1:
                    logger.info(f"\nEpoch {index+1}: eval_f1 improved from {best_f1} to {eval_results['f1']}")
                    output_dir = os.path.join(args.output_dir, "best_model")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), os.path.join(
                        output_dir, "pytorch_model.bin"))
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    best_f1 = eval_results['f1']
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, processor, model, prefix="", show_entity_info=False):
    metric = SeqEntityScore(processor.idx2label)
    eval_dataset = NER_dataset(load_and_cache_examples(args, processor, data_type='dev'),
                               max_seq_len=args.eval_max_seq_len)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "input_mask": batch[1],
                      "labels": batch[2], 'input_lens': batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['input_mask'])
        if args.n_gpu > 1:
            # mean() to average on multi-gpu parallel evaluating
            tmp_eval_loss = tmp_eval_loss.mean()
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = inputs['input_lens'].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(processor.idx2label[out_label_ids[i][j]])
                    temp_2.append(processor.idx2label[tags[i][j]])
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    info = "Eval results" + "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    if show_entity_info:
        logger.info("***** Entity results %s *****", prefix)
        for key in sorted(entity_info.keys()):
            logger.info("******* %s results ********" % key)
            info = "-".join([f' {key}: {value:.4f} ' for key,
                            value in entity_info[key].items()])
            logger.info(info)
    return results


def predict(args, processor, model, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = NER_dataset(load_and_cache_examples(args, processor, data_type='test'),
                               args.eval_max_seq_len)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(
        test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []
    output_predict_file = os.path.join(
        pred_output_dir, prefix, "test_prediction.json")

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "input_mask": batch[1],
                      "labels": None, 'input_lens': batch[3]}
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['input_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()
        preds = tags[0]
        label_entities = get_entity_bios(preds, processor.idx2label)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join([processor.idx2label[x] for x in preds])
        json_d['entities'] = label_entities
        results.append(json_d)
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    output_submit_file = os.path.join(
        pred_output_dir, prefix, "test_submit.json")
    test_text = []
    with open(os.path.join(args.data_path, "test.json"), 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {}
        json_d['id'] = x['id']
        json_d['label'] = {}
        entities = y['entities']
        words = list(x['text'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file, test_submit)
