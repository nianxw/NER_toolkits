import os
import logging
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from models import ner_model, fgm
from tool_utils import data_helper, util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def data_to_cuda(batch):
    return_lists = []
    for t in batch:
        if isinstance(t, torch.Tensor):
            return_lists += [t.cuda()]
        else:
            return_lists += [t]
    return return_lists


def batch_forward(batch, model):
    # input_ids/input_mask/segment_ids/labels/input_len
    encoder_output = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])
    loss = -1*model.crf(emissions=encoder_output, tags=batch[3], mask=batch[1])
    return encoder_output, loss


def train(tokenizer, config, args, train_data_set, eval_data_set=None):
    # 获取数据
    num_train_optimization_steps = int(
        len(train_data_set) / args.train_batch_size) * args.epochs
    train_data_loader = DataLoader(dataset=train_data_set,
                                   batch_size=args.train_batch_size,
                                   collate_fn=data_helper.collate_fn)

    # 构建模型
    steps = 0
    model = ner_model.BertCrfForNer.from_pretrained(args.init_checkpoint, config=config)

    if args.do_adv:
        fgm_model = fgm.FGM(model)  # 定义对抗训练模型

    if args.use_cuda:
        # model = nn.DataParallel(model)
        model.cuda()

    # prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=1e-8)

    warmup_step = num_train_optimization_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_step, num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    log_loss = 0.0

    best_f1 = 0.0

    begin_time = time.time()
    model.train()
    for epoch in range(args.epochs):
        for batch in train_data_loader:
            steps += 1
            if args.use_cuda:
                batch = data_to_cuda(batch)

            _, loss = batch_forward(batch, model)
            loss.backward()

            # 对抗训练
            if args.do_adv:
                fgm_model.attack()
                _, loss_adv = batch_forward(batch, model)
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
                        log_loss / args.log_steps,
                        used_time / args.log_steps,
                    ),
                )
                begin_time = time.time()
                log_loss = 0.0

            if args.do_eval and steps % args.eval_step == 0:
                eval_info = evaluate(args, eval_data_set, model)
                eval_f1 = eval_info['f1']
                if eval_f1 > best_f1:
                    logging.info('save model: %s' % os.path.join(
                        args.save_path, 'model_%d.bin' % (steps)))
                    torch.save(model.state_dict(), os.path.join(args.save_path, 'model_best.bin'))
                    best_f1 = eval_f1
                    logging.info('best f1: %.4f' % best_f1)
    logging.info('final best f1: %.4f' % best_f1)


def evaluate(args, eval_data_set, model):
    metric = util.SeqEntityScore(args.id2label)

    eval_sampler = SequentialSampler(eval_data_set)
    eval_data_loader = DataLoader(dataset=eval_data_set, sampler=eval_sampler,
                                  batch_size=args.eval_batch_size, collate_fn=data_helper.collate_fn)

    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    eval_loss = 0.0
    eval_step = 0
    for batch_eval in eval_data_loader:
        if args.use_cuda:
            batch_eval = data_to_cuda(batch_eval)
        with torch.no_grad():
            eval_logits, tmp_eval_loss = batch_forward(batch_eval, model)
            tags = model.crf.decode(eval_logits, batch_eval[1])

        eval_step += 1
        eval_loss += tmp_eval_loss.item()
        out_label_ids = batch_eval[3].cpu().numpy().tolist()
        input_lens = batch_eval[4].cpu().numpy().tolist()
        sentences = batch_eval[6]
        tags = tags.squeeze().cpu().numpy().tolist()

        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            s = sentences[i]
            for j, m in enumerate(label):
                # 去除[CLS]和[SEP]
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(args.id2label[tags[i][j]])
    ave_loss = eval_loss / eval_step
    eval_info, entity_info = metric.result()
    logging.info("eval res\tave loss:%.4f\tPrecision: %.4f\tRecall: %.4f\tF1: %.4f" % (ave_loss, eval_info['acc'], eval_info['recall'], eval_info['f1']))
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    model.train()
    return eval_info


def predict(args, test_data_set, model):
    test_dataloader = DataLoader(test_data_set,
                                 batch_size=1,
                                 collate_fn=data_helper.collate_fn)

    logger.info("***** Running prediction %s *****")
    logger.info("  Num examples = %d", len(test_data_set))
    logger.info("  Batch size = %d", 1)

    predict_res = {}
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch_test in tqdm(enumerate(test_dataloader)):
        model.eval()
        if args.use_cuda:
            batch_test = data_to_cuda(batch_test)
        with torch.no_grad():
            test_logits = model(input_ids=batch_test[0], attention_mask=batch_test[1], token_type_ids=batch_test[2])
            tags = model.crf.decode(test_logits, batch_test[1])
            tags = tags.squeeze().cpu().numpy().tolist()
        preds = tags[1:-1]  # [CLS]XXXX[SEP]
        label_entities = util.get_entity_bios(preds, args.id2label)
        predict_res[batch_test[5][0]] = label_entities
    return predict_res
