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

    def __init__(self, input_file, tokenizer, max_seq_len, type2description, shuffle=True):
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.max_seq_len = max_seq_len
        self.type2description = type2description
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
                    labels = line.get('labels', [])
                    new_label = defaultdict(list)
                    if labels:
                        for label in labels:
                            new_label[label[2]].append(label[:2])

                    token_b = self.tokenizer.tokenize(sentence)
                    if len(token_b) <= 1:
                        continue
                    for entity_type, entity_des in self.type2description.items():
                        token_a = self.tokenizer.tokenize(entity_des)
                        q_len = len(token_a)
                        tokens = [self.cls_token] + token_a + [self.sep_token] + token_b + [self.sep_token]
                        tokens_len = len(tokens)

                        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                        segment_ids = [0] * (q_len + 2) + [1] * (len(token_b) + 1)
                        input_mask = [1] * tokens_len

                        left_label = [0] * tokens_len
                        right_label = [0] * tokens_len
                        label_mask = [0] * (q_len + 2) + [1] * len(token_b) + [0]  # 只求当前句子范围的loss

                        if entity_type in new_label:
                            entity_pos = new_label[entity_type]
                            for pos in entity_pos:
                                left_pos = pos[0] + q_len + 2
                                right_pos = pos[1] + q_len + 1
                                left_label[left_pos] = 1
                                right_label[right_pos] = 1

                        # pad
                        pad_len = self.max_seq_len - tokens_len
                        input_ids += [0] * pad_len
                        segment_ids += [0] * pad_len
                        input_mask += [0] * pad_len
                        left_label += [0] * pad_len
                        right_label += [0] * pad_len
                        label_mask += [0] * pad_len

                        data.append(
                            {
                                'text_id': text_id,
                                'input_ids': input_ids,
                                'segment_ids': segment_ids,
                                'input_mask': input_mask,
                                'left_label': left_label,
                                'right_label': right_label,
                                'label_mask': label_mask,
                                'q_len': q_len + 2,
                                'input_len': tokens_len,
                                'entity_type': entity_type,
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
    batch_label_mask = []  # 计算loss的mask，与 input_mask 不同
    batch_left_label = []
    batch_right_label = []
    batch_q_len = []
    batch_input_len = []
    batch_entity_type = []

    for i in range(len(examples)):
        example = examples[i]

        batch_text_id.append(example['text_id'])
        batch_input_ids.append(example['input_ids'])
        batch_segment_ids.append(example['segment_ids'])
        batch_input_mask.append(example['input_mask'])
        batch_label_mask.append(example['label_mask'])
        batch_left_label.append(example['left_label'])
        batch_right_label.append(example['right_label'])
        batch_q_len.append(example['q_len'])
        batch_input_len.append(example['input_len'])
        batch_entity_type.append(example['entity_type'])

    max_len = max(batch_input_len)
    batch_input_ids = torch.tensor(np.array(batch_input_ids)[:, : max_len], dtype=torch.long)
    batch_segment_ids = torch.tensor(np.array(batch_segment_ids)[:, : max_len], dtype=torch.long)
    batch_input_mask = torch.tensor(np.array(batch_input_mask)[:, : max_len], dtype=torch.float)
    batch_label_mask = torch.tensor(np.array(batch_label_mask)[:, : max_len], dtype=torch.float)
    batch_left_label = torch.tensor(np.array(batch_left_label)[:, : max_len], dtype=torch.long)
    batch_right_label = torch.tensor(np.array(batch_right_label)[:, : max_len], dtype=torch.long)

    return_list = []
    # input_ids/segment_ids/input_mask/label_mask/left_label/right_label/
    return_list += [batch_input_ids]
    return_list += [batch_segment_ids]
    return_list += [batch_input_mask]
    return_list += [batch_label_mask]
    return_list += [batch_left_label]
    return_list += [batch_right_label]

    # eval/pred
    return_list += [batch_text_id]
    return_list += [batch_q_len]
    return_list += [batch_input_len]
    return_list += [batch_entity_type]
    return return_list
