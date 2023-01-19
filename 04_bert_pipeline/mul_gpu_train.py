import argparse
import logging
import os
import json
import random

import numpy as np
import torch
import pandas as pd

from transformers import BertConfig, BertModel

from models import Basic_model
from tool_utils import data_helper, train_helper_mul_gpu, util


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# export CUDA_VISIBLE_DEVICES=

def main():
    parser = argparse.ArgumentParser()

    # 1. 训练和测试数据路径
    parser.add_argument("--train_path", default='train_data/train_v1.json', type=str, help="Path to data.")
    parser.add_argument("--dev_path", default='train_data/dev.json', type=str, help="Path to data.")
    parser.add_argument("--predict_path", default='test_data/11.3_predict_bert.json', type=str, help="Path to data.")


    # 2. 预训练模型路径
    parser.add_argument("--vocab_file", default="bert-base-chinese/vocab.txt", type=str, help="Init vocab to resume training from.")
    parser.add_argument("--config_path", default="bert-base-chinese/config.json", type=str, help="Init config to resume training from.")
    parser.add_argument("--init_checkpoint", default="bert-base-chinese/pytorch_model.bin", type=str, help="Init checkpoint to resume training from.")

    # 3. 保存模型
    parser.add_argument("--save_path", default="./check_points", type=str, help="Path to save checkpoints.")
    parser.add_argument("--load_path", default="./check_points/model_best.bin", type=str, help="Path to load checkpoints.")

    # 训练和测试参数
    parser.add_argument("--do_train", default=True, type=bool, help="Whether to perform training.")
    parser.add_argument("--do_eval", default=False, type=bool, help="Whether to perform evaluation on test data set.")
    parser.add_argument("--do_predict", default=False, type=bool, help="Whether to perform evaluation on test data set.")
    parser.add_argument("--do_adv", default=False, type=bool)


    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--epochs", default=3, type=int, help="Number of epoches for fine-tuning.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total examples' number in batch for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Total examples' number in batch for eval.")
    parser.add_argument("--max_seq_len", default=280, type=int, help="Number of words of the longest seqence.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate used to train with warmup.")
    parser.add_argument("--warmup_proportion", default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")

    # 多卡训练
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda")
    parser.add_argument("--log_steps",
                        type=int,
                        default=100,
                        help="The steps interval to print loss.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=2000,
                        help="The steps interval to save model.")
    parser.add_argument("--eval_step",
                        type=int,
                        default=200,
                        help="The steps interval to print loss.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if args.local_rank == -1 or not args.use_cuda:
        print(torch.cuda.is_available())
        args.device = torch.device(f"cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
        print(args.device)
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {}, n_gpu: {}, distributed training: {}".format(args.device, args.n_gpu, args.local_rank != -1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    bert_tokenizer = util.CNerTokenizer.from_pretrained(args.vocab_file)
    bert_config = BertConfig.from_pretrained(args.config_path)
    bert_config.num_labels = args.num_labels

    type2id = bert_tokenizer.get_label(args.train_path)
    args.type2id = type2id
    args.id2type = {v: k for k, v in type2id.items()}

    # 获取数据
    train_dataset = None
    if args.do_train:
        logger.info("loading train dataset")
        train_dataset = data_helper.NER_dataset(args.train_path, bert_tokenizer,
                                                args.max_seq_len, args.type2id)

    if args.do_train:
        logging.info("Start training !")
        train_helper_mul_gpu.train(bert_tokenizer, bert_config, args, train_dataset)

    if args.do_predict:
        logging.info("Start predicting !")
        ner_model = Basic_model.ner.from_pretrained(args.load_path)
        logging.info("Checkpoint: %s have been loaded!" % (args.load_path))

        if args.use_cuda:
            ner_model.cuda()

        predict_res = train_helper_mul_gpu.predict(args, bert_tokenizer, ner_model)



if __name__ == "__main__":
    main()
