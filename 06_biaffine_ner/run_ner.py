import argparse
import logging
import os
import json
import random

import numpy as np
import torch
from transformers import BertConfig

from models import biaffine_ner
from tool_utils import data_helper, train_helper, util


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# export CUDA_VISIBLE_DEVICES=
def main():
    parser = argparse.ArgumentParser()

    # 1. 训练和测试数据路径
    parser.add_argument("--data_dir", default='./data/cluener', type=str, help="Path to data.")

    # 2. 预训练模型路径
    parser.add_argument("--vocab_file", default="data/pretrain/vocab.txt", type=str, help="Init vocab to resume training from.")
    parser.add_argument("--config_path", default="data/pretrain/config.json", type=str, help="Init config to resume training from.")
    parser.add_argument("--init_checkpoint", default="data/pretrain/pytorch_model.bin", type=str, help="Init checkpoint to resume training from.")

    # 3. 保存模型
    parser.add_argument("--save_path", default="./check_points", type=str, help="Path to save checkpoints.")
    parser.add_argument("--load_path", default=None, type=str, help="Path to load checkpoints.")

    # 训练和测试参数
    parser.add_argument("--do_train", default=True, type=bool, help="Whether to perform training.")
    parser.add_argument("--do_eval", default=True, type=bool, help="Whether to perform evaluation on test data set.")
    parser.add_argument("--do_predict", default=False, type=bool, help="Whether to perform evaluation on test data set.")
    parser.add_argument("--do_adv", default=True, type=bool)

    parser.add_argument("--epochs", default=20, type=int, help="Number of epoches for fine-tuning.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total examples' number in batch for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Total examples' number in batch for eval.")
    parser.add_argument("--max_seq_len", default=256, type=int, help="Number of words of the longest seqence.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate used to train with warmup.")
    parser.add_argument("--warmup_proportion", default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")

    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda")
    parser.add_argument("--log_steps",
                        type=int,
                        default=20,
                        help="The steps interval to print loss.")
    parser.add_argument("--eval_step",
                        type=int,
                        default=200,
                        help="The steps interval to print loss.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if args.use_cuda:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model_path_postfix = ''
    if args.do_adv:
        model_path_postfix += '_adv'

    args.save_path = os.path.join(args.save_path, model_path_postfix)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    args.label2id = {
        'NONE': 0,
        'address': 0,
        'book': 1,
        'company': 2,
        'game': 3,
        'government': 4,
        'movie': 5,
        'name': 6,
        'organization': 7,
        'position': 8,
        'scene': 9
    }
    args.id2label = {v: k for k, v in args.label2id.items()}

    bert_tokenizer = util.CNerTokenizer.from_pretrained(args.vocab_file)
    bert_config = BertConfig.from_pretrained(args.config_path, num_labels=len(args.label2id))

    # 获取数据
    train_dataset = None
    eval_dataset = None
    train_dataset = None
    if args.do_train:
        logger.info("loading train dataset")
        train_dataset = data_helper.NER_dataset(os.path.join(args.data_dir, 'train.json'), bert_tokenizer,
                                                args.max_seq_len, args.label2id)

    if args.do_eval:
        logger.info("loading eval dataset")
        eval_dataset = data_helper.NER_dataset(os.path.join(args.data_dir, 'dev.json'), bert_tokenizer,
                                               args.max_seq_len, args.label2id,
                                               shuffle=False)

    if args.do_predict:
        logger.info("loading test dataset")
        test_dataset = data_helper.NER_dataset(os.path.join(args.data_dir, 'test.json'), bert_tokenizer,
                                               args.max_seq_len, args.label2id,
                                               shuffle=False)

    if args.do_train:
        logging.info("Start training !")
        train_helper.train(bert_tokenizer, bert_config, args, train_dataset, eval_dataset)

    if not args.do_train and args.do_eval:
        logging.info("Start evaluating !")
        biaffine_model = biaffine_ner.Biaffine_NER(bert_config)
        biaffine_model.load_state_dict(torch.load(args.load_path))
        logging.info("Checkpoint: %s have been loaded!" % (args.load_path))

        if args.use_cuda:
            biaffine_model.cuda()
        train_helper.evaluate(args, eval_dataset, biaffine_model)

    if args.do_predict:
        logging.info("Start predicting !")
        biaffine_model = biaffine_ner.Biaffine_NER(bert_config)
        biaffine_model.load_state_dict(torch.load(args.load_path))
        logging.info("Checkpoint: %s have been loaded!" % (args.load_path))

        if args.use_cuda:
            biaffine_model.cuda()
        predict_res = train_helper.predict(args, test_dataset, biaffine_model)


if __name__ == "__main__":
    main()
