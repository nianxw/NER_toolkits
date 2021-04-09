import os
import json
import logging
import pickle
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class NER_dataset(Dataset):

    def __init__(self, input_file, tokenizer, max_seq_len, type2id, shuffle=True):
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.max_seq_len = max_seq_len
        self.type2id = type2id
        self.data = self.load_data(input_file)

    def load_data(self, input_file):
        cache_file = input_file.replace('.json', '_cache.pkl')
        if os.path.exists(cache_file):
            logger.info("loading data from cache file: %s" % cache_file)
            return pickle.load(open(cache_file, 'rb'))
        else:
            logger.info("loading data from input file: %s" % input_file)
            with open(input_file, 'r', encoding='utf8') as f:
                index = 1
                data = []
                for line in f:
                    if index % 1000 == 0:
                        logger.info("%d examples have been loaded" % index)
                    line = json.loads(line.strip())
                    if self.shuffle:
                        text_id_pre_fix = 'train'
                    else:
                        text_id_pre_fix = 'dev'
                    text_id = line.get('text_id', text_id_pre_fix+str(index))
                    sentence = line['text']
                    labels = line.get('label', [])

                    token_a = self.tokenizer.tokenize(sentence)
                    if len(token_a) <= 1:
                        continue
                    # id
                    tokens = [self.cls_token] + token_a + [self.sep_token]
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    tokens_len = len(tokens)
                    segment_ids = [0] * tokens_len
                    input_mask = [1] * tokens_len
                    label_mask = [0] + [1] * (tokens_len - 2) + [0]

                    # pad
                    pad_len = self.max_seq_len - tokens_len
                    input_ids += [0] * pad_len
                    segment_ids += [0] * pad_len
                    input_mask += [0] * pad_len
                    label_mask += [0] * pad_len

                    # label
                    biaffine_label = np.zeros([self.max_seq_len, self.max_seq_len])
                    if labels:
                        for label in labels:
                            left = label[0] + 1
                            right = label[1]
                            entity_type_id = self.type2id[label[2]]
                            biaffine_label[left][right] = entity_type_id

                    data.append(
                        {
                            'text_id': text_id,
                            'input_ids': input_ids,
                            'segment_ids': segment_ids,
                            'input_mask': input_mask,
                            'biaffine_label': biaffine_label,
                            'label_mask': label_mask,
                            'input_len': tokens_len,

                        }
                    )
                    index += 1
            if self.shuffle:
                np.random.shuffle(data)
            pickle.dump(data, open(cache_file, 'wb'))
            return data

    def __getitem__(self, idx):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    examples = data

    # input
    batch_text_id = []
    batch_input_ids = []
    batch_segment_ids = []
    batch_input_mask = []  # 输入的mask
    batch_biaffine_label = []
    batch_label_mask = []
    batch_input_len = []

    for i in range(len(examples)):
        example = examples[i]
        batch_text_id.append(example['text_id'])
        batch_input_ids.append(example['input_ids'])
        batch_segment_ids.append(example['segment_ids'])
        batch_input_mask.append(example['input_mask'])
        batch_biaffine_label.append(example['biaffine_label'])
        batch_label_mask.append(example['label_mask'])
        batch_input_len.append(example['input_len'])

    max_len = max(batch_input_len)
    batch_input_ids = torch.tensor(np.array(batch_input_ids)[:, : max_len], dtype=torch.long)
    batch_segment_ids = torch.tensor(np.array(batch_segment_ids)[:, : max_len], dtype=torch.long)
    batch_input_mask = torch.tensor(np.array(batch_input_mask)[:, : max_len], dtype=torch.float)
    batch_biaffine_label = torch.tensor(np.array(batch_biaffine_label)[:, : max_len, : max_len], dtype=torch.long)
    batch_label_mask = torch.tensor(np.array(batch_label_mask)[:, : max_len], dtype=torch.long)
    return_list = []
    # input_ids/segment_ids/input_mask/biaffine_label/label_mask
    return_list += [batch_input_ids]
    return_list += [batch_segment_ids]
    return_list += [batch_input_mask]
    return_list += [batch_biaffine_label]
    return_list += [batch_label_mask]

    # eval/pred
    return_list += [batch_text_id]
    return_list += [batch_input_len]
    return return_list
