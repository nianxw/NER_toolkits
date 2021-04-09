import os
import json
from transformers import BertTokenizer


class CNerTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens

    def get_label(self, data_dir):
        label_path = os.path.join(data_dir, 'label.json')
        if os.path.exists(label_path):
            return json.load(open(label_path, 'r', encoding='utf8'))
        else:
            label_set = set()
            with open(os.path.join(data_dir, 'train.json'), 'r') as fr:
                for line in fr:
                    line = json.loads(line.strip())
                    labels = line['label']
                    for l in labels:
                        label_set.add(l[-1])
            label2id = {}
            for l in label_set:
                label2id[l] = len(label2id)
            label2id['O'] = len(label2id)
            json.dump(label2id, open(label_path, 'w', encoding='utf8'), ensure_ascii=False)
            return label2id


def binary_decode1(left_pos, right_pos):
    start_pos = []
    end_pos = []
    for l_pos in left_pos:
        r_pos = right_pos[right_pos > l_pos]
        if len(r_pos) > 0:
            r_pos = r_pos[0]
            start_pos.append(l_pos)
            end_pos.append(r_pos)
    return start_pos, end_pos


def binary_decode2(left, right):
    start_pos, end_pos = binary_decode1(left, right)
    res = [[start_pos[i], end_pos[i]] for i in range(len(start_pos))]
    res_len = len(res)
    new_res = [1] * res_len

    if not res:
        return [], []

    s = res[0][0]
    e = res[0][1]

    for i in range(1, res_len):
        cur = res[i]
        if e == cur[1]:
            new_res[i-1] = 0
        s = cur[0]
        e = cur[1]

    final_res = [res[i] for i in range(res_len) if new_res[i] == 1]
    return_s = []
    return_e = []
    for i in range(len(final_res)):
        return_s.append(final_res[i][0])
        return_e.append(final_res[i][1])
    return return_s, return_e