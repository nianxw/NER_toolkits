import argparse


def get_argparse():
    parser = argparse.ArgumentParser()

    # 路径参数
    parser.add_argument("--data_path", default="./datasets/", type=str,
                        help="Input: tran/dev/test data path")
    parser.add_argument("--output_dir", default="./outputs/", type=str,
                        help="Output: predictions and checkpoints path")
    parser.add_argument("--model_path",
                        default=None,
                        type=str, help="已训练的模型，可在此基础上继续训练")

    # 模型参数
    parser.add_argument("--emb_size", default=128, type=int)
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.15, type=float)
    parser.add_argument("--fc_dropout", default=0.3, type=float)
    parser.add_argument("--after_norm", default=True, type=bool)
    parser.add_argument("--scale", default=False, type=bool)

    # 训练参数
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--do_predict", default=True, action="store_true")

    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--train_max_seq_len", default=128, type=int)
    parser.add_argument("--eval_max_seq_len", default=512, type=int)

    parser.add_argument("--learning_rate", default=0.0007, type=float)
    parser.add_argument("--crf_learning_rate", default=0.0007, type=float)
    parser.add_argument("--momentum_rate", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--optim", default='adam', type=str, help="sgd/adam")

    parser.add_argument("--warmup_rate", default=0.01, type=float)
    parser.add_argument("--log_steps", default=100, type=int)
    parser.add_argument("--save_steps", default=200, type=int)

    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--gpu", default=None, type=str)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--seed", default=321, type=int)

    return parser
