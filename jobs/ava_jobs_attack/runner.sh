#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

task_name=$1
dataset=$2
target_model_train_dataset=$3
model=$4
max_seq_len=$5
batch_size=$6
toolchain=$7
attack=$8
max_queries=$9

python -u scripts/attack.py \
    --task_name $task_name \
    --dataset_name $dataset \
    --target_model_train_dataset $target_model_train_dataset \
    --model_name $model \
    --model_max_seq_len $max_seq_len \
    --model_batch_size $batch_size \
    --attack_toolchain $toolchain \
    --attack_name $attack \
    --attack_max_queries $max_queries
