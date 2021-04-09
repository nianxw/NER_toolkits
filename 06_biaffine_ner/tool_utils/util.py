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


def biaffine_decode(pred, threshold, keep_nested=True):
    if not pred:
        return []
    pred_len = len(pred)

    pred_flag = [1] * pred_len
    for i in range(pred_len):
        p = pred[i][-1]
        if p < threshold:
            pred_flag[i] = 0

    left, right, entity_type, entity_prob = pred[0]
    for i in range(1, pred_len):
        cur_left, cur_right, cur_entity_type, cur_entity_prob = pred[i]
        if cur_left == left:
            if not keep_nested:
                if cur_entity_prob > entity_prob:
                    pred_flag[i-1] = 0
                    left, right, entity_type, entity_prob = pred[i]
                else:
                    pred_flag[i] = 0
                    left, right, entity_type, entity_prob = pred[i-1]
            else:
                left, right, entity_type, entity_prob = pred[i]
        else:
            if cur_left <= right:
                if cur_right <= right and keep_nested:
                    left, right, entity_type, entity_prob = pred[i-1]
                    continue
                else:
                    if cur_entity_prob > entity_prob:
                        pred_flag[i-1] = 0
                        left, right, entity_type, entity_prob = pred[i]
                    else:
                        pred_flag[i] = 0
                        left, right, entity_type, entity_prob = pred[i-1]
            else:
                left, right, entity_type, entity_prob = pred[i]
    res = []
    for i in range(pred_len):
        left, right, entity_type = pred[i][:-1]
        if pred_flag[i]:
            res.append([left, right + 1, entity_type])
    return res



def easy_decode():
    # 思路是对所有实体候选片段按照概率值大小进行降序排序
    # 如果保留嵌套实体，则只将出现交叉的片段对，去除概率较小的那个，靠后的片段
    # 如果不保留嵌套实体，则除上述交叉之外，还应按概率大小删除包含关系的片段对
    pass


