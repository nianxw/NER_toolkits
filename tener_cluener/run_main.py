import os
import time
import torch
from models.TENER import NER_model
from utils.ner_argparse import get_argparse
from utils.common import init_logger, logger, seed_everything
from processors.ner_seq import CluenerProcessor
from utils.train_helper import train, evaluate, predict


def main():
    args = get_argparse().parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    log_file = os.path.join(args.output_dir, 'tener-{}.log'.format(time_))
    init_logger(log_file=log_file)
    if args.gpu and args.use_cuda:
        device = torch.device("cuda:%s" % args.gpu)
        args.n_gpu = 1
    elif args.local_rank == -1 or args.use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)  # torch.Tensor分配到的设备对象
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    seed_everything(args.seed)

    ner_processor = CluenerProcessor(args.data_path)
    model = NER_model(ner_processor, args)
    model.to(args.device)
    if args.model_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'pytorch_model.bin')))

    if args.do_train:
        train(args, ner_processor, model)

    if args.do_eval:
        evaluate(args, ner_processor, model, show_entity_info=True)

    if args.do_predict:
        predict(args, ner_processor, model)


if __name__ == "__main__":
    main()