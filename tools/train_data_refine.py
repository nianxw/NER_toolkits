import os
import json
from collections import defaultdict, Counter

# 对基于词典的远程监督数据进行修正
# 8折交叉

base_path = ''   # 训练集的多个预测结果
src_train_data_path = ''  # 原始训练集
new_train_data_path = ''  # 输出的新训练集


src_train_data = json.load(open(src_train_data_path, 'r', encoding='utf8'))  # 原始训练集路径
file_list = os.listdir(base_path)

pred_list = []
for f in file_list:
    file_path = os.path.join(base_path, f)
    pred_list.append(json.load(open(file_path, 'r', encoding='utf8')))
len_ = len(pred_list)


ensemble_res = {}
for i in range(len(pred_list[0])):
    tmp_res = []
    ensemble = []
    for p in pred_list:
        ensemble += p[i]['entities']
    d = Counter(ensemble)
    text_id = pred_list[0][i]['id']
    ensemble_res[text_id] = d


# 新增每个类型的实体
add_entity = defaultdict(list)
# 被修改的实体
modify_entity = []
# 被删除的实体
delete_entity = []


# 新增：对于少样本类别实体，频次>=2都可以保留加入实体词典中，其他类别要>4
# 修改：对某个实体，其类别在预测时若和标签不同，且频次>4，则修改其类别
# 删除：对于多样本类别实体，标签在预测时一次都没出现，则删除

big_e_type = ['RIV', 'LAK', 'LOC', 'RES']
big_e_type1 = ['RIV', 'LAK', 'LOC']

max_entity_len = 0  # 最大实体长度
min_entity_len = 100  # 最小实体长度
total_src_label_nums = 0  # 原始数据label数目
total_new_label_nums = 0  # 新数据label数目
new_train_data = []
for src_d in src_train_data:
    text_id = src_d['id']
    text = src_d['text']
    new_labels = []
    src_labels = src_d['entities']
    pred_labels = ensemble_res[text_id]

    # 统计预测实体的频次
    pred_word_to_type = {}   # {'word': [type, n]}
    for e, v in pred_labels.items():
        try:
            e_word, e_type = e.split('-')
            if e_word not in pred_word_to_type:
                pred_word_to_type[e_word] = defaultdict(int)
            pred_word_to_type[e_word][e_type] += v
        except:
            continue

    for e_word, e_type_info in pred_word_to_type.items():
        type_count = []
        for e_type, count in e_type_info.items():
            type_count.append([e_type, count])
        type_count.sort(key=lambda x: x[1])
        pred_word_to_type[e_word] = type_count[0]

    # 修改和删除
    for src_e in src_labels:
        src_e_word, src_e_type = src_e.split('-')
        max_entity_len = max(max_entity_len, len(src_e_word))
        min_entity_len = min(min_entity_len, len(src_e_word))
        if src_e_word in pred_word_to_type:
            e_type, count = pred_word_to_type[src_e_word]
            if src_e_type != e_type and count > 4:
                new_e = src_e_word + '-' + e_type
                new_labels.append(new_e)  # 修改类别
                modify_entity.append(src_e + '-->' + new_e)
            else:
                new_labels.append(src_e)
        else:
            new_labels.append(src_e)
            # if src_e_type not in big_e_type1:
            #     new_labels.append(src_e)
            # else:
            # delete_entity.append(src_e)  # 删除实体

    # 新增
    for e_word, e_type_info in pred_word_to_type.items():
        e_type, e_count = e_type_info
        if e_type not in big_e_type:
            threshold = 8
        else:
            threshold = 8
        if e_count >= threshold:
            if len(e_word) >= 10 or '，' in e_word or '）' in e_word or '（' in e_word:
                continue
            pred_label = e_word + '-' + e_type
            if pred_label not in new_labels:
                new_labels.append(pred_label)
                add_entity[e_type].append(e_word)

    total_src_label_nums += len(src_labels)
    total_new_label_nums += len(new_labels)
    new_train_data.append({
        'id': text_id,
        'text': text,
        'entities': new_labels,
    })

json.dump(new_train_data, open(new_train_data_path, 'w', encoding='utf8'), ensure_ascii=False)

for k, v in add_entity.items():
    print('type: %s\tnums: %d' % (k, len(list(set(v)))))
    print(list(set(v)))
# print(add_entity)

print('*************************')
print(modify_entity)
print('修改的实体数目%d' % len(modify_entity))
print(delete_entity)
print('删除的实体数目：%d' % len(delete_entity))
print('实体词最大长度：%d' % max_entity_len)
print('实体词最小长度：%d' % min_entity_len)

print('原始label数目：%d' % total_src_label_nums)
print('新label数目：%d' % total_new_label_nums)