#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate textattack-0.2.11

dataset=$1
model=$2
optimizer=$3
max_seq_len=$4
lr=$5
batch_size=$6
epochs=$7
weight_decay=$8
max_norm=$9

python3 scripts/train.py \
    --dataset $dataset \
    --model $model \
    --optimizer $optimizer \
    --max_seq_len $max_seq_len \
    --lr $lr \
    --batch_size $batch_size \
    --epochs $epochs \
    --weight_decay $weight_decay \
    --max_norm $max_norm
