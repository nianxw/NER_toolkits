import os
import json
import logging
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel

from models import span_type, fgm, Basic_model
from tool_utils import data_helper, util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def data_to_cuda(batch, args):
    return_lists = []
    for t in batch:
        if isinstance(t, torch.Tensor):
            return_lists += [t.to(args.device)]
        else:
            return_lists += [t]
    return return_lists


def batch_forward(batch, model, span_loss_fc, type_loss_fc, cate_loss_fc, config):
    # input_ids/input_mask/label_mask/left_label/right_label/left_pos/right_pos/type_label/type_label_mask
    left_logits, right_logits, cate_logits, entity_type_logits, entity_type_probs = model(batch)
    # 计算 span loss
    left_logits = left_logits.view(-1)
    left_label = batch[3].float().view(-1)
    right_logits = right_logits.view(-1)
    right_label = batch[4].float().view(-1)

    label_mask = batch[2].float().view(-1)
    loss_left = span_loss_fc(left_logits, left_label) * label_mask
    loss_right = span_loss_fc(right_logits, right_label) * label_mask
    loss_span = torch.mean(loss_left) + torch.mean(loss_right)

    # 计算 type loss
    entity_type_logits = entity_type_logits.view(-1, 10)
    type_label = batch[7].view(-1)
    type_label_mask = batch[8].view(-1)
    loss_type = type_loss_fc(entity_type_logits, type_label) * type_label_mask
    loss_type = torch.mean(loss_type)

    # 计算 cate loss
    cate_logits = cate_logits.view(-1, config.num_labels)
    cate_label = batch[9].view(-1)
    loss_cate = cate_loss_fc(cate_logits, cate_label)
    loss_cate = torch.mean(loss_cate)

    loss = loss_span + loss_cate + loss_type

    return loss, left_logits, right_logits, entity_type_probs, cate_logits


def train(tokenizer, config, args, train_data_set):
    # 获取数据
    num_train_optimization_steps = int(
        len(train_data_set) / args.train_batch_size) * args.epochs

    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        train_sampler = DistributedSampler(train_data_set)
    else:
        train_sampler = RandomSampler(train_data_set)
    
    train_data_loader = DataLoader(dataset=train_data_set,
                                   batch_size=args.train_batch_size,
                                   collate_fn=data_helper.collate_fn,
                                   sampler=train_sampler)

    # 构建模型
    steps = 0
    model = Basic_model.ner(config)
    model.from_pretrained('bert-base-chinese')
    model.to(args.device)

    if args.do_adv:
        fgm_model = fgm.FGM(model)  # 定义对抗训练模型
                                                        

    # prepare optimizer
    parameters_to_optimize = list(model.parameters())
    optimizer = AdamW(parameters_to_optimize,
                      lr=args.learning_rate)

    warmup_step = num_train_optimization_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_step, num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    log_loss = 0.0

    span_loss_fc = nn.BCELoss(reduce=False)
    type_loss_fc = nn.CrossEntropyLoss(reduce=False)
    cate_loss_fc = nn.CrossEntropyLoss(reduce=False)

    model.train()
    begin_time = time.time()

    for epoch in range(args.epochs):
        for batch in train_data_loader:
            steps += 1
            if args.use_cuda:
                batch = data_to_cuda(batch, args)

            loss = batch_forward(batch, model, span_loss_fc, type_loss_fc, cate_loss_fc, config)[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()

            # 对抗训练
            if args.do_adv:
                fgm_model.attack()
                loss_adv = batch_forward(batch, model, span_loss_fc, type_loss_fc, cate_loss_fc, config)[0]
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm_model.restore()  # 恢复embedding参数

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            log_loss += loss.data.item()

            if steps % args.log_steps == 0:
                end_time = time.time()
                used_time = end_time - begin_time
                logger.info(
                    "epoch: %d, progress: %d/%d, ave loss: %f, speed: %f s/step" %
                    (
                        epoch, steps, num_train_optimization_steps,
                        loss,  # log_loss / args.log_steps,
                        used_time / args.log_steps,
                    ),
                )
                begin_time = time.time()
                log_loss = 0.0

        if args.local_rank in [-1, 0]:
            args.output_dir = Path(args.save_path)
            output_dir = args.output_dir / f'lm-checkpoint-{epoch}'
            if not output_dir.exists():
                output_dir.mkdir()
            # save model
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(str(output_dir))
            # torch.save(args, str(output_dir / 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

            # torch.save(optimizer.state_dict(), str(output_dir / "optimizer.bin"))
            # save config
            output_config_file = output_dir / "config.json"
            with open(str(output_config_file), 'w') as f:
                f.write(model_to_save.config.to_json_string())
            # save vocab
            tokenizer.save_vocabulary(output_dir)


def predict(args, tokenizer, model_list):
    bert_model, span_model, type_model = model_list
    bert_model.eval()
    span_model.eval()
    type_model.eval()
    save_path = '/'.join(args.load_path.split('/')[:-1])
    output = open(os.path.join(save_path, args.predict_path.split('/')[-1] + '.res'), 'w', encoding='utf8')

    predict_res = defaultdict(list)
    with torch.no_grad():
        with open(args.predict_path, 'r', encoding='utf8') as f:
            for line in tqdm(f):
                line = json.loads(line.strip())
                text_id = line['nid']
                sentence = line['text']
                app_id = line.get('app_id', '123')
                tokens = tokenizer.tokenize(sentence)
                if len(tokens) > args.max_seq_len - 2:
                    tokens = tokens[: args.max_seq_len - 2]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.tensor([input_ids]).long()
                if args.use_cuda:
                    input_ids = input_ids.cuda()

                encoder_output, pooled_output = bert_model(input_ids=input_ids)[:2]
                left_logits, right_logits, cate_logits = span_model(encoder_output=encoder_output, pooled_output=pooled_output)
                left_logits = left_logits.cpu().numpy()
                right_logits = right_logits.cpu().numpy()
                cate_logits = torch.softmax(cate_logits, dim=-1).cpu().numpy()
                cate_logits = cate_logits.tolist()[0]
                cate_score = max(cate_logits)
                cate_pred_label = cate_logits.index(cate_score)
                left_pos, right_pos = np.where(left_logits[0] > 0.5)[0], np.where(right_logits[0] > 0.5)[0]  # 返回的是索引
                start_pos, end_pos = util.binary_decode2(left_pos, right_pos)


                # if start_pos:
                #     start_pos = torch.tensor(np.array([[_ for _ in start_pos]]), device=input_ids.device)
                #     end_pos = torch.tensor(np.array([[_ for _ in end_pos]]), device=input_ids.device)
                #     _, entity_type_probs = type_model(encoder_output=encoder_output,
                #                                       s_pos=start_pos,
                #                                       e_pos=end_pos)
                #     _, type_preds = torch.max(entity_type_probs, -1)
                #     type_preds = type_preds.cpu().tolist()[0]
                #     start_pos_cpu = start_pos.data.cpu().tolist()[0]
                #     end_pos_cpu = end_pos.data.cpu().tolist()[0]

                #     for s, e, t in zip(start_pos_cpu, end_pos_cpu, type_preds):
                #         predict_res[text_id].append([s-1, e, t])

                #     pred_label = []
                #     for s, e, t in zip(start_pos_cpu, end_pos_cpu, type_preds):
                #         span_type = args.id2type[t]
                #         span_text = sentence[s-1: e]
                #         pred_label.append([s-1, e, span_type, span_text])
                #     line['label'] = pred_label

                # if cate_score >= 0.5:
                # if cate_score >= 0.5:
                if cate_pred_label != 0:
                    line['score'] = str(cate_score)
                    line['cate'] = str(cate_pred_label)
                    output.write(json.dumps(line, ensure_ascii=False) + '\n')
                    output.flush()

    output.close()
    return predict_res
