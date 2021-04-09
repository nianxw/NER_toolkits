import os
import json
import logging
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel

from models import span_type, fgm
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


def batch_forward(batch, model_list, span_loss_fc):
    bert_model, span_model = model_list

    # input_ids/segment_ids/input_mask/label_mask/left_label/right_label/
    encoder_output = bert_model(input_ids=batch[0], attention_mask=batch[2].float(), token_type_ids=batch[1])[0]
    left_logits, right_logits = span_model(encoder_output=encoder_output)

    # 计算 span loss
    left_logits = left_logits.view(-1)
    left_label = batch[4].float().view(-1)
    right_logits = right_logits.view(-1)
    right_label = batch[5].float().view(-1)

    label_mask = batch[3].float().view(-1)
    loss_left = span_loss_fc(left_logits, left_label) * label_mask
    loss_right = span_loss_fc(right_logits, right_label) * label_mask
    loss_span = torch.mean(loss_left) + torch.mean(loss_right)

    return loss_span, left_logits, right_logits


def train(tokenizer, config, args, train_data_set, eval_data_set=None):
    # 获取数据
    num_train_optimization_steps = int(
        len(train_data_set) / args.train_batch_size) * args.epochs
    train_data_loader = DataLoader(dataset=train_data_set,
                                   batch_size=args.train_batch_size,
                                   collate_fn=data_helper.collate_fn)

    # 构建模型
    steps = 0
    bert_model = BertModel.from_pretrained(args.init_checkpoint, config=config)
    span_model = span_type.EntitySpan(config=config)

    if args.do_adv:
        fgm_model = fgm.FGM(bert_model)  # 定义对抗训练模型

    if args.use_cuda:
        # model = nn.DataParallel(model)
        bert_model.cuda()
        span_model.cuda()

    # prepare optimizer
    parameters_to_optimize = list(bert_model.parameters()) + list(span_model.parameters())
    optimizer = AdamW(parameters_to_optimize,
                      lr=args.learning_rate)

    warmup_step = num_train_optimization_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_step, num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model_list = [bert_model, span_model]

    log_loss = 0.0

    best_f1 = 0.0

    span_loss_fc = nn.BCELoss(reduce=False)
    begin_time = time.time()

    bert_model.train()
    span_model.train()
    for epoch in range(args.epochs):
        for batch in train_data_loader:
            steps += 1
            if args.use_cuda:
                batch = data_to_cuda(batch)

            loss = batch_forward(batch,
                                 model_list,
                                 span_loss_fc)[0]

            loss.backward()

            # 对抗训练
            if args.do_adv:
                fgm_model.attack()
                loss_adv = batch_forward(batch, model_list, span_loss_fc)[0]
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

            if args.do_eval and steps % args.eval_step == 0:
                eval_f1 = evaluate(args, eval_data_set, model_list)
                if eval_f1 > best_f1:
                    logging.info('save model: %s' % os.path.join(
                        args.save_path, 'model_%d.bin' % epoch))

                    torch.save(
                        {
                            'bert_state_dict': bert_model.state_dict(),
                            'span_state_dict': span_model.state_dict(),
                        },
                        os.path.join(args.save_path, 'model_best.bin'))
                    best_f1 = eval_f1
                    logging.info('best f1: %.4f' % best_f1)


def evaluate(args, eval_data_set, model_list):
    eval_data_loader = DataLoader(dataset=eval_data_set, batch_size=1, collate_fn=data_helper.collate_fn)
    bert_model, span_model = model_list
    bert_model.eval()
    span_model.eval()

    logger.info("***** Running evaluating *****")
    logger.info("  Num examples = %d", len(eval_data_set))
    logger.info("  Batch size = 1")

    precision_num = 0.0
    recall_num = 0.0
    correct_num = 0.0

    pred_res = defaultdict(list)
    with torch.no_grad():
        for batch_eval in eval_data_loader:
            if args.use_cuda:
                batch_eval = data_to_cuda(batch_eval)
            encoder_output = bert_model(input_ids=batch_eval[0], attention_mask=batch_eval[2].float(), token_type_ids=batch_eval[1])[0]
            left_logits, right_logits = span_model(encoder_output=encoder_output)
            left_logits = left_logits.cpu().numpy()  # [1, seq_len, 1]
            right_logits = right_logits.cpu().numpy()
            left_pos, right_pos = np.where(left_logits[0] > 0.5)[0], np.where(right_logits[0] > 0.5)[0]  # 返回的是索引
            text_id = batch_eval[6][0]
            q_len = batch_eval[7][0]
            entity_type = batch_eval[9][0]
            pred_res[text_id].extend(util.binary_decode2(left_pos, right_pos, q_len, entity_type))

    label_res = {}
    index = 1
    with open(os.path.join(args.data_dir, 'dev.json'), 'r', encoding='utf8') as f:
        for line in f:
            line = json.loads(line.strip())
            labels = line['label']
            label_res[line.get('text_id', 'dev'+str(index))] = labels
            index += 1

    for k, v in label_res.items():
        precision_num += len(pred_res[k])
        recall_num += len(v)
        correct_num += len([_ for _ in pred_res[k] if _ in v])

    bert_model.train()
    span_model.train()

    precison = correct_num / (precision_num + 1e-5)
    recall = correct_num / recall_num
    f1 = 2*precison*recall / (precison+recall + 1e-5)

    logging.info("eval res\tPrecision: %.4f\tRecall: %.4f\tF1: %.4f" % (precison, recall, f1))
    return f1


def predict(args, test_data_set, model_list):
    bert_model, span_model = model_list
    bert_model.eval()
    span_model.eval()

    test_data_loader = DataLoader(dataset=test_data_set, batch_size=1, collate_fn=data_helper.collate_fn)
    logger.info("***** Running predicting *****")
    logger.info("  Num examples = %d", len(test_data_loader))
    logger.info("  Batch size = 1")

    pred_res = defaultdict(list)
    with torch.no_grad():
        for batch_pred in tqdm(test_data_loader):
            if args.use_cuda:
                batch_pred = data_to_cuda(batch_pred)
            encoder_output = bert_model(input_ids=batch_pred[0], attention_mask=batch_pred[2].float(), token_type_ids=batch_pred[1])[0]
            left_logits, right_logits = span_model(encoder_output=encoder_output)
            left_logits = left_logits.cpu().numpy()  # [1, seq_len, 1]
            right_logits = right_logits.cpu().numpy()
            left_pos, right_pos = np.where(left_logits[0] > 0.5)[0], np.where(right_logits[0] > 0.5)[0]  # 返回的是索引
            text_id = batch_pred[6][0]
            q_len = batch_pred[7][0]
            entity_type = batch_pred[9][0]
            pred_res[text_id].extend(util.binary_decode2(left_pos, right_pos, q_len, entity_type))
    return pred_res
