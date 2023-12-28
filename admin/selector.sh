#!/bin/bash
cd ..
set -e
set -v

BASE_DIR=/

TASK_NAME=slb1217

# TRAIN_PATH=./data/train_public.txt
# VALID_PATH=./data/sighan15_2.txt
TRAIN_PATH=./data/train_realise.txt
VALID_PATH=./data/sighan15_realise.txt
LR=2e-5
AMP=True
PARALLEL=True
NUM_EPOCH=30
RESUME=True
SEED=17

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 1256 start.py \
	--cuda_num 4 \
	--num_workers 4 \
	--task_name $TASK_NAME \
	--train_file $TRAIN_PATH \
	--val_file $VALID_PATH \
	--n_epoch $NUM_EPOCH \
	--batch_size 32 \
	--lr $LR \
	--parallel $PARALLEL \
