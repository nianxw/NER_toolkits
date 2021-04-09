import os
import json
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class NER_dataset(Dataset):

    def __init__(self, input_file, tokenizer, max_seq_len, label2id, shuffle=True):
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label2id = label2id
        self.data = self.load_data(input_file)

    def load_data(self, input_file):
        cache_file = input_file.replace('.json', '_cache.pkl')
        if os.path.exists(cache_file):
            logger.info("loading data from cache file: %s" % cache_file)
            return pickle.load(open(cache_file, 'rb'))
        else:
            logger.info("loading data from input file: %s" % input_file)
            with open(input_file, 'r', encoding='utf8') as f:
                i = 1
                data = []
                for line in f:
                    if i % 5000 == 0:
                        logger.info("%d examples have been loaded" % i)
                    line = json.loads(line.strip())

                    text_id = line.get('text_id')
                    sentence = line.get('text')
                    label_entities = line.get('label', None)

                    # 中文-按字划分
                    words = list(sentence)
                    labels = ['O'] * len(words)
                    if label_entities:
                        for entity_info in label_entities:
                            left = entity_info[0]
                            right = entity_info[1]
                            entity_type = entity_info[2]
                            if left + 1 == right:
                                labels[left] = 'S-' + entity_type
                            else:
                                labels[left] = 'B-' + entity_type
                                labels[left + 1: right] = ['I-' + entity_type] * (right - left - 1)

                    # tokenize
                    tokens = self.tokenizer.tokenize(words)
                    label_ids = [self.label2id[x] for x in labels]
                    if len(tokens) > self.max_seq_len:
                        tokens = tokens[: self.max_seq_len]
                        label_ids = label_ids[: self.max_seq_len]

                    # id化
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)
                    label_ids = label_ids

                    # pad
                    input_len = len(input_ids)
                    padding_length = self.max_seq_len - input_len
                    input_ids += [0] * padding_length
                    input_mask += [0] * padding_length
                    label_ids += [0] * padding_length

                    assert len(input_ids) == self.max_seq_len
                    assert len(input_mask) == self.max_seq_len
                    assert len(label_ids) == self.max_seq_len

                    data.append(
                        {   
                            'text_id': text_id,
                            'input_ids': input_ids,
                            'input_mask': input_mask,
                            'label_ids': label_ids,
                            'input_len': input_len,
                            'sentence': sentence,
                        }
                    )
                    i += 1
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
    batch_text_ids = []
    batch_input_ids = []
    batch_input_mask = []  # 输入的mask
    batch_label_ids = []
    batch_input_len = []
    batch_sentence = []

    for i in range(len(examples)):
        example = examples[i]

        batch_text_ids.append(example['text_id'])
        batch_input_ids.append(example['input_ids'])
        batch_input_mask.append(example['input_mask'])
        batch_label_ids.append(example['label_ids'])
        batch_input_len.append(example['input_len'])
        batch_sentence.append(example['sentence'])

    max_len = max(batch_input_len)
    batch_input_ids = torch.tensor(np.array(batch_input_ids)[:, : max_len], dtype=torch.long)
    batch_input_mask = torch.tensor(np.array(batch_input_mask)[:, : max_len], dtype=torch.float)
    batch_label_ids = torch.tensor(np.array(batch_label_ids)[:, : max_len], dtype=torch.long)
    batch_input_len = torch.tensor(batch_input_len, dtype=torch.long)
    return batch_input_ids, batch_input_mask, batch_label_ids, batch_input_len, batch_text_ids, batch_sentence
