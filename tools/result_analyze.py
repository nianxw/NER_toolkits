import os
import json
from collections import defaultdict, Counter

index = 1


# 计算每个类别对应的精度和召回
def compute(origin, found, right):
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def result(origins, founds, rights):
    class_info = {}
    origin_counter = Counter([x.split('-')[1] for x in origins])
    found_counter = Counter([x.split('-')[1] for x in founds])
    right_counter = Counter([x.split('-')[1] for x in rights])
    for type_, count in origin_counter.items():
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = compute(origin, found, right)
        class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
    return class_info


# 获取原训练数据label分布
src_train_path = ''  # 原数据
src_entity = {}
src_label = json.load(open(src_train_path, 'r', encoding='utf8'))
for src_d in src_label:
    src_entity[src_d['id']] = src_d['entities']


# 获取预测数据label分布
pred_train_path = ''  # 预测结果
pred_entity = {}
pred = json.load(open(pred_train_path, 'r', encoding='utf8'))
for pred_d in pred:
    pred_entity[pred_d['id']] = pred_d['entities']


origins = []
founds = []
rights = []
for k, v in src_entity.items():
    origins.extend(v)
    p = pred_entity.get(k, [])
    founds.extend(p)
    rights.extend([_ for _ in p if _ in v])


# 计算精度和召回
print('precision: %.4f\trecall: %.4f\tf1: %.4f' % (compute(len(origins), len(founds), len(rights))))

class_info = result(origins, founds, rights)
for k, v in class_info.items():
    print('type: %s\tprecision: %.4f\trecall: %.4f\tf1: %.4f' % (k, v['acc'], v['recall'], v['f1']))