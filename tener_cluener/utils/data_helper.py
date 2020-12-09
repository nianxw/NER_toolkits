import os
import torch
from torch.utils.data import Dataset
from utils.common import logger


def load_and_cache_examples(args, processor, data_type='train'):
    # Load data features from cache or dataset file
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    cached_examples_file = os.path.join(args.data_path, f'cached_crf-{data_type}')
    if os.path.exists(cached_examples_file):
        logger.info("Loading features from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_path)
        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
        else:
            examples = processor.get_test_examples()
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_examples_file)
            torch.save(examples, str(cached_examples_file))
    if args.local_rank == 0:
        torch.distributed.barrier()
    return examples


class NER_dataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.data = data
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        example = self.data[idx]
        token_ids = example['token_ids']
        tag_ids = example['tag_ids']
        data_len = len(token_ids)

        if data_len > self.max_seq_len:
            input_id = token_ids[: self.max_seq_len]
            input_mask = [1]*self.max_seq_len
            label_id = tag_ids[: self.max_seq_len]
        else:
            input_id = token_ids + [0]*(self.max_seq_len-data_len)
            input_mask = [1]*data_len + [0]*(self.max_seq_len-data_len)
            label_id = tag_ids + [0]*(self.max_seq_len-data_len)
        return input_id, input_mask, label_id, data_len

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    all_input_ids, all_input_mask, all_labels, all_lens = zip(*batch)
    max_len = max(all_lens)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)[:, :max_len]
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)[:, :max_len]
    all_labels = torch.tensor(all_labels, dtype=torch.long)[:, :max_len]
    all_lens = torch.tensor(all_lens, dtype=torch.long)
    return all_input_ids, all_input_mask, all_labels, all_lens
