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
                i = 1
                data = []
                for line in f:
                    if i % 1000 == 0:
                        logger.info("%d examples have been loaded" % i)
                    line = json.loads(line.strip())

                    sentence = line['text']
                    labels = line['label']

                    # 模型输入
                    input_ids = self.tokenize_sentence(sentence)
                    input_mask = [1]*len(input_ids)
                    input_len = len(input_ids)

                    left_label = [0]*input_len
                    right_label = [0]*input_len

                    type_label = []
                    left_pos = []
                    right_pos = []

                    entity_positions = []

                    for label in labels:
                        left = label[0] + 1
                        right = label[1]
                        entity_type = label[2]
                        if right > input_len:
                            continue
                        entity_positions.append([label[0], label[1]])
                        left_label[left] = 1
                        right_label[right] = 1
                        left_pos.append(left)
                        right_pos.append(right)
                        type_label.append(self.type2id[entity_type])

                    label_mask = [1]*len(input_ids)

                    data.append(
                        {
                            'input_ids': input_ids,
                            'input_mask': input_mask,
                            'label_mask': label_mask,
                            'left_label': left_label,
                            'right_label': right_label,
                            'left_pos': left_pos,
                            'right_pos': right_pos,
                            'type_label': type_label,
                        }
                    )
                    i += 1
            if self.shuffle:
                np.random.shuffle(data)
            pickle.dump(data, open(cache_file, 'wb'))
            return data

    def tokenize_sentence(self, sentecne):
        tokens = self.tokenizer.tokenize(sentecne)
        if len(tokens) > (self.max_seq_len - 2):
            tokens = tokens[: self.max_seq_len - 2]
        tokens = [self.cls_token] + tokens + [self.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_ids

    def __getitem__(self, idx):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)


def pad_seq(insts, return_seq_mask=False):
    return_list = []

    max_len = max(len(inst) for inst in insts)

    # input ids
    inst_data = np.array(
        [inst + list([0] * (max_len - len(inst))) for inst in insts],
    )
    return_list += [inst_data.astype("int64")]

    if return_seq_mask:
        # input mask
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        return_list += [input_mask_data.astype("float32")]

    return return_list


def collate_fn(data):
    examples = data

    # input
    batch_input_ids = []
    batch_input_mask = []  # 输入的mask
    batch_label_mask = []  # 计算loss的mask，此时二者可能不同。因为引入关键词后，其对于的loss不参与计算。
    batch_left_label = []
    batch_right_label = []
    batch_left_pos = []
    batch_right_pos = []
    batch_type_label = []

    for i in range(len(examples)):
        example = examples[i]

        batch_input_ids.append(example['input_ids'])
        batch_input_mask.append(example['input_mask'])
        batch_label_mask.append(example['label_mask'])
        batch_left_label.append(example['left_label'])
        batch_right_label.append(example['right_label'])
        batch_left_pos.append(example['left_pos'])
        batch_right_pos.append(example['right_pos'])
        batch_type_label.append(example['type_label'])

    return_list = []
    # seq pad
    return_list += pad_seq(batch_input_ids)
    return_list += pad_seq(batch_input_mask)
    return_list += pad_seq(batch_label_mask)
    return_list += pad_seq(batch_left_label)
    return_list += pad_seq(batch_right_label)
    return_list += pad_seq(batch_left_pos)
    return_list += pad_seq(batch_right_pos)
    return_list += pad_seq(batch_type_label, True)

    # input_ids/input_mask/label_mask/left_label/right_label/left_pos/right_pos/type_label/type_label_mask
    return_list = [torch.tensor(batch_data) for batch_data in return_list]
    return return_list
