import argparse
import os
import torch
import torch.distributed as dist

from xiatian import L
from mectec.csc.train import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name',
                        help='train task name',
                        default='xiatian')
    parser.add_argument('--train_set',
                        help='Path to the train data',
                        default='./data/train_realise.txt')
    parser.add_argument('--dev_set',
                        help='Path to the dev data',
                        default='./data/sighan15_realise.txt')
    parser.add_argument('--test_set',
                        help='Path to the test data',
                        default='./data/sighan15_realise.txt')
    parser.add_argument('--vocab_path',
                        help='Path to the model vocabulary directory.',
                        default='./vocabulary/')
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the batch.',
                        default=32)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                        '(all longer will be truncated)',
                        default=256)
    parser.add_argument('--start_epoch',
                        type=int,
                        help='The start epoch for training model.',
                        default=0)
    parser.add_argument('--n_epoch',
                        type=int,
                        help='The number of epoch for training model.',
                        default=10)
    parser.add_argument('--lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=2e-5)
    parser.add_argument('--transformer_model',
                        help='Name of the transformer model.',
                        default='./pretrain/chinese-roberta-wwm-ext')
    parser.add_argument("--model_type",                        
                        type=str,
                        default='pinyin',
                        help="model type")
    parser.add_argument('--model_dir',
                        help='Path to the gec model dir',
                        default='./model/megec_dev')
    parser.add_argument('--pinyin_path',
                        type=str,
                        help='The path of the pinyin',
                        default="")
    parser.add_argument('--additional_confidence',
                        type=int,
                        help='',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=int,
                        help='',
                        default=0)
    parser.add_argument('--cuda_num',
                        type=int,
                        help='The number of cudas',
                        default=1)
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=1)
    parser.add_argument("--local-rank", 
                        default=os.getenv('LOCAL_RANK', -1), 
                        type=int, 
                        help="node rank for distributed training")
    parser.add_argument("--nproc_per_node",
                        default=1,
                        type=int,
                        help="node rank for distributed training")
    parser.add_argument("--amp",
                        default=False,
                        type=bool,
                        help="Whether to use mixed precision")
    parser.add_argument("--parallel",
                        default=False,
                        type=bool,
                        help="Whether to use parallel")
    parser.add_argument("--last_ckpt_file",
                        default='',
                        help="last checkpoint file")
    parser.add_argument("--resume",
                        default=False,
                        type=bool,
                        help="Whether to resume training")
    parser.add_argument("--seed",
                        default=17,
                        type=int,
                        help="random seed")
    parser.add_argument('--label_loss_weight',
                        type=float,
                        help='Set initial learning rate.',
                        default=0.5)

    args = parser.parse_args()
                    
    if args.local_rank == -1:
        args.local_rank = os.getenv('LOCAL_RANK', -1)
                    
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    L.info(f'nproc_per_node: {args.nproc_per_node}')
    L.info(f'args: {args}')
    trainer = Trainer(
        task_name=args.task_name,
        model_name="best_model.pth",
        device=device,
        max_len=args.max_len,
        start_epoch=args.start_epoch,
        epoch_num=args.n_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout_rate=0.1,
        base_model=args.transformer_model,
        correct_f1_weight=0.5,
        label_loss_weight=args.label_loss_weight,
        last_ckpt_file=args.last_ckpt_file,
        parallel=args.parallel,
        additional_confidence=args.additional_confidence,
        additional_del_confidence=args.additional_del_confidence,
        amp=args.amp,
        local_rank=args.local_rank,
        cuda_num=args.cuda_num,
        resume=args.resume,
        seed=args.seed,
        model_type=args.model_type,
        train_path = args.train_set, 
        val_path=args.dev_set,
        test_path=args.test_set)
    trainer.train(val_step=1000)
