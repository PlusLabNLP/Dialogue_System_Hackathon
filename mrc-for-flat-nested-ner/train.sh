#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 

FOLDER_PATH=/nas/home/zixiliu/mrc-for-flat-nested-ner
#${FOLDER_PATH}/config/zh_bert.json
CONFIG_PATH=/nas/home/zixiliu/BERT/base_uncased/uncased_L-12_H-768_A-12/bert_config.json
#/PATH-TO-BERT_MRC-DATA/zh_ontonotes4
DATA_PATH=/nas/home/zixiliu/mrc-for-flat-nested-ner/ontonotes-release-5.0/ontonote5
#/PATH-TO-BERT-CHECKPOINTS/chinese_L-12_H-768_A-12
BERT_PATH=/nas/home/zixiliu/BERT/base_uncased/uncased_L-12_H-768_A-12
#/PATH-TO-SAVE-MODEL-CKPT 
EXPORT_DIR=/nas/home/zixiliu/mrc-for-flat-nested-ner/checkpoint
data_sign=en_onto 
entity_sign=flat

export PYTHONPATH=${FOLDER_PATH}
CUDA_VISIBLE_DEVICES=0 python3 ${FOLDER_PATH}/run/train_bert_mrc.py \
--config_path ${CONFIG_PATH} \
--data_dir ${DATA_PATH} \
--bert_model ${BERT_PATH} \
--output_dir ${EXPORT_DIR} \
--entity_sign ${entity_sign} \
--data_sign ${data_sign} \
--n_gpu 1 \
--export_model True \
--dropout 0.3 \
--checkpoint 600 \
--max_seq_length 100 \
--train_batch_size 16 \
--dev_batch_size 16 \
--test_batch_size 16 \
--learning_rate 8e-6 \
--weight_start 1.0 \
--weight_end 1.0 \
--weight_span 1.0 \
--num_train_epochs 10 \
--seed 2333 \
--warmup_proportion -1 \
--gradient_accumulation_steps 1