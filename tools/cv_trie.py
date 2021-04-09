import json
from sklearn.model_selection import KFold

train_data = []
test_data = []


class Trie:
    def __init__(self, with_type=False):
        self.with_type = with_type
        self.root = {}

    def insert(self, word, entity_type=None):
        root = self.root
        for c in word:
            if c not in root:
                root[c] = {}
            root = root[c]
        root['is_end'] = True
        if entity_type:
            root['entity_type'] = entity_type

    def search(self, sentence):
        i = 0
        res = []
        while i < len(sentence):
            root = self.root
            while i < len(sentence) and sentence[i] not in root:
                i += 1
            j = i
            tmp = []  # 保存查询到的词的索引
            while j < len(sentence) and sentence[j] in root:
                root = root[sentence[j]]
                j += 1
                if root.get('is_end', False):
                    if self.with_type:
                        tmp.append([i, j, root.get('entity_type')])
                    else:
                        tmp.append([i, j])
            if tmp:
                # 如何查询到了多个词，则取最长作为结果
                res.append(tmp[-1])
                i = tmp[-1][1]
            else:
                i += 1
        return res


# 有交叉时，保留长的词
f1_data = json.load(open('./data/src/bmes_train.json', 'r', encoding='utf8'))

label_map = set()

for line in f1_data:
    text_id = line['id']
    text = line['text']
    labels = []
    entity = line['entities']
    trie_ = Trie(True)
    for e in entity:
        e, e_type = e.split('-')
        label_map.add(e_type)
        trie_.insert(e, e_type)
    res = trie_.search(text)
    train_data.append({
        'text_id': text_id,
        'text': text,
        'labels': res
    })


final_label_map = []
label_map = list(label_map)
for name in label_map:
    final_label_map += ['B-'+name, 'I-'+name, 'S-'+name]
final_label_map += ['O', '[START]', '[END]']

final_label_map_dict = {}
for i in final_label_map:
    final_label_map_dict[i] = len(final_label_map_dict)
json.dump(final_label_map_dict, open('./train_data_random_state_5/label_map.json', 'w', encoding='utf8'))


kfold = KFold(n_splits=8, shuffle=True, random_state=2)
# kfold = KFold(n_splits=8, shuffle=True, random_state=5)

i = 0
for train, test in kfold.split(train_data):
    with open('./train_data_random_state_5/train_%d.json' % i, 'w', encoding='utf8') as f:
        for index_train in train:
            f.write(json.dumps(train_data[index_train], ensure_ascii=False)+'\n')

    with open('./train_data_random_state_5/dev_%d.json' % i, 'w', encoding='utf8') as f:
        for index_test in test:
            f.write(json.dumps(train_data[index_test], ensure_ascii=False)+'\n')

    i += 1


test_file = open('./train_data_random_state_5/test.json', 'w', encoding='utf8')
f2_data = json.load(open('./data/src/bmes_test.json', 'r', encoding='utf8'))

for line in f2_data:
    test_file.write(json.dumps({'text_id': line['id'], 'text': line['text']}, ensure_ascii=False)+'\n')

test_file.close()









