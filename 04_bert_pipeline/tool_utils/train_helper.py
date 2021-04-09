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


def batch_forward(batch, model_list, span_loss_fc, type_loss_fc):
    bert_model, span_model, type_model = model_list

    # input_ids/input_mask/label_mask/left_label/right_label/left_pos/right_pos/type_label/type_label_mask
    encoder_output = bert_model(input_ids=batch[0], attention_mask=batch[1].float())[0]
    left_logits, right_logits = span_model(encoder_output=encoder_output)
    entity_type_logits, entity_type_probs = type_model(encoder_output=encoder_output,
                                                       s_pos=batch[5],
                                                       e_pos=batch[6])

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

    loss = loss_span + loss_type

    return loss, left_logits, right_logits, entity_type_probs


def train(tokenizer, config, args, train_data_set):
    # 获取数据
    num_train_optimization_steps = int(
        len(train_data_set) / args.train_batch_size) * args.epochs
    train_data_loader = DataLoader(dataset=train_data_set,
                                   batch_size=args.train_batch_size,
                                   collate_fn=data_helper.collate_fn)

    # 构建模型
    steps = 0
    bert_model = BertModel.from_pretrained(args.init_checkpoint, config=config)
    span_model = span_type.EntitySpan(config)
    type_model = span_type.EntityType()

    if args.do_adv:
        fgm_model = fgm.FGM(bert_model)  # 定义对抗训练模型

    if args.use_cuda:
        # model = nn.DataParallel(model)
        bert_model.cuda()
        span_model.cuda()
        type_model.cuda()

    # prepare optimizer
    parameters_to_optimize = list(bert_model.parameters()) + list(span_model.parameters()) + list(type_model.parameters())
    optimizer = AdamW(parameters_to_optimize,
                      lr=args.learning_rate)

    warmup_step = num_train_optimization_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_step, num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model_list = [bert_model, span_model, type_model]

    log_loss = 0.0

    best_f1 = 0.0

    span_loss_fc = nn.BCELoss(reduce=False)
    type_loss_fc = nn.CrossEntropyLoss(reduce=False)

    bert_model.train()
    span_model.train()
    type_model.train()
    begin_time = time.time()
    for epoch in range(args.epochs):
        for batch in train_data_loader:
            steps += 1
            if args.use_cuda:
                batch = data_to_cuda(batch)

            loss = batch_forward(batch, model_list,
                                 span_loss_fc, type_loss_fc)[0]

            loss.backward()

            # 对抗训练
            if args.do_adv:
                fgm_model.attack()
                loss_adv = batch_forward(batch, model_list, span_loss_fc, type_loss_fc)[0]
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
                eval_f1 = evaluate(args, tokenizer, model_list)
                if eval_f1 > best_f1:
                    logging.info('save model: %s' % os.path.join(
                        args.save_path, 'model_%d.bin' % epoch))

                    torch.save(
                        {
                            'bert_state_dict': bert_model.state_dict(),
                            'span_state_dict': span_model.state_dict(),
                            'type_state_dict': type_model.state_dict()
                        },
                        os.path.join(args.save_path, 'model_best.bin'))
                    best_f1 = eval_f1
                    logging.info('best f1: %.4f' % best_f1)
    logging.info('final best f1: %.4f' % best_f1)


def evaluate(args, tokenizer, model_list):
    bert_model, span_model, type_model = model_list
    bert_model.eval()
    span_model.eval()
    type_model.eval()

    precision_num = 0.0
    recall_num = 0.0
    correct_num = 0.0

    with torch.no_grad():
        with open(os.path.join(args.data_dir, 'dev.json'), 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line.strip())
                sentence = line['text']
                tokens = tokenizer.tokenize(sentence)
                if len(tokens) > args.max_seq_len - 2:
                    tokens = tokens[: args.max_seq_len - 2]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.tensor([input_ids]).long()
                if args.use_cuda:
                    input_ids = input_ids.cuda()

                encoder_output = bert_model(input_ids=input_ids)[0]
                left_logits, right_logits = span_model(encoder_output=encoder_output)
                left_logits = left_logits.cpu().numpy()
                right_logits = right_logits.cpu().numpy()
                left_pos, right_pos = np.where(left_logits[0] > 0.5)[0], np.where(right_logits[0] > 0.5)[0]  # 返回的是索引
                start_pos, end_pos = util.binary_decode2(left_pos, right_pos)

                tmp_entity_info = []
                if start_pos:
                    start_pos = torch.tensor(np.array([[_ for _ in start_pos]]), device=input_ids.device)
                    end_pos = torch.tensor(np.array([[_ for _ in end_pos]]), device=input_ids.device)
                    _, entity_type_probs = type_model(encoder_output=encoder_output,
                                                      s_pos=start_pos,
                                                      e_pos=end_pos)
                    _, type_preds = torch.max(entity_type_probs, -1)
                    type_preds = type_preds.cpu().tolist()[0]
                    start_pos_cpu = start_pos.data.cpu().tolist()[0]
                    end_pos_cpu = end_pos.data.cpu().tolist()[0]

                    for s, e, t in zip(start_pos_cpu, end_pos_cpu, type_preds):
                        tmp_entity_info.append([s-1, e, args.id2type[t]])

                labels = line['label']
                precision_num += len(tmp_entity_info)
                recall_num += len(labels)
                correct_num += len([_ for _ in tmp_entity_info if _ in labels])

    bert_model.train()
    span_model.train()
    type_model.train()

    precison = correct_num / (precision_num + 1e-5)
    recall = correct_num / recall_num
    f1 = 2*precison*recall / (precison+recall + 1e-5)
    logging.info("eval res\tPrecision: %.4f\tRecall: %.4f\tF1: %.4f" % (precison, recall, f1))
    return f1


def predict(args, tokenizer, model_list):
    bert_model, span_model, type_model = model_list
    bert_model.eval()
    span_model.eval()
    type_model.eval()

    predict_res = defaultdict(list)
    with torch.no_grad():
        with open(os.path.join(args.data_dir, 'test.json'), 'r', encoding='utf8') as f:
            for line in tqdm(f):
                line = json.loads(line.strip())
                text_id = line['text_id']
                sentence = line['text']
                tokens = tokenizer.tokenize(sentence)
                if len(tokens) > args.max_seq_len - 2:
                    tokens = tokens[: args.max_seq_len - 2]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.tensor([input_ids]).long()
                if args.use_cuda:
                    input_ids = input_ids.cuda()

                encoder_output = bert_model(input_ids=input_ids)[0]
                left_logits, right_logits = span_model(encoder_output=encoder_output)
                left_logits = left_logits.cpu().numpy()
                right_logits = right_logits.cpu().numpy()
                left_pos, right_pos = np.where(left_logits[0] > 0.5)[0], np.where(right_logits[0] > 0.5)[0]  # 返回的是索引
                start_pos, end_pos = util.binary_decode2(left_pos, right_pos)

                if start_pos:
                    start_pos = torch.tensor(np.array([[_ for _ in start_pos]]), device=input_ids.device)
                    end_pos = torch.tensor(np.array([[_ for _ in end_pos]]), device=input_ids.device)
                    _, entity_type_probs = type_model(encoder_output=encoder_output,
                                                      s_pos=start_pos,
                                                      e_pos=end_pos)
                    _, type_preds = torch.max(entity_type_probs, -1)
                    type_preds = type_preds.cpu().tolist()[0]
                    start_pos_cpu = start_pos.data.cpu().tolist()[0]
                    end_pos_cpu = end_pos.data.cpu().tolist()[0]

                    for s, e, t in zip(start_pos_cpu, end_pos_cpu, type_preds):
                        predict_res[text_id].append([s-1, e, t])

    return predict_res
