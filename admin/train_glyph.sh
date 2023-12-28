#!/bin/bash
cd ..
set -e
set -v

BASE_DIR=/
# train: realise, val: realise, task_name likes bestXXXX
# train: public, val: realise, task_name likes wishXXXX
TASK_NAME=final_glyph

#TRAIN_PATH=./data/train_public.txt
# VALID_PATH=./data/sighan15_2.txt
# TEST_PATH=./data/sighan15_2.txt

TRAIN_PATH=./data/train_realise.txt
VALID_PATH=./data/sighan15_realise.txt
TEST_PATH=./data/sighan15_realise.txt

#BASE_MODEL=./pretrain/structbert
BASE_MODEL=./pretrain/chinese-roberta-wwm-ext
VOCAB_PATH=./vocabulary/
pinyin_path=./vocabulary/pinyin_vocab.txt
char_emb_path=./vocabulary/hanzi_ViT.pkl
LR=2e-5
AMP=False
PARALLEL=True
NUM_EPOCH=30
RESUME=True
seed=17
LABEL_LOSS_WEIGHT=0.5
model_type=glyph


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
	--nproc-per-node=4 --master_port 1193 main.py \
	--cuda_num 4 \
	--num_workers 4 \
	--train_set $TRAIN_PATH \
	--dev_set $VALID_PATH \
	--test_set $TEST_PATH \
	--transformer_model  $BASE_MODEL\
	--task_name $TASK_NAME \
	--model_type $model_type \
    --vocab_path $VOCAB_PATH \
	--n_epoch $NUM_EPOCH \
	--batch_size 32 \
	--lr $LR \
	--max_len 150 \
	--amp $AMP \
	--parallel $PARALLEL \
	--seed $seed \
	--pinyin_path $pinyin_path \
