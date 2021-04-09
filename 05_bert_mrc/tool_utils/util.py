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


def binary_decode1(left, right, q_len, entity_type):
    res = []
    for l_pos in left:
        if l_pos < q_len:
            continue
        r_pos = right[right >= l_pos]
        if len(r_pos) > 0:
            r_pos = r_pos[0]
            res.append([l_pos - q_len, r_pos - q_len + 1, entity_type])
    return res


def binary_decode2(left, right, q_len, entity_type):
    res = binary_decode1(left, right, q_len, entity_type)
    res_len = len(res)
    new_res = [1] * res_len

    if not res:
        return []

    s = res[0][0]
    e = res[0][1]

    for i in range(1, res_len):
        cur = res[i]
        if e == cur[1]:
            new_res[i-1] = 0
        s = cur[0]
        e = cur[1]

    return [res[i] for i in range(res_len) if new_res[i] == 1]
