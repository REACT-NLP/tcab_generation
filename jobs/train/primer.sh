dataset=$1
model=$2
optimizer=$3
max_seq_len=$4
lr=$5
batch_size=$6
epochs=$7
weight_decay=$8
max_norm=$9
mem=${10}
time=${11}
partition=${12}

job_name=T_${dataset}_${model}
sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --gres=gpu:1 \
       --job-name=$job_name \
       --output=jobs/logs/train/$job_name \
       --error=jobs/errors/train/$job_name \
       jobs/train/runner.sh $dataset $model $optimizer $max_seq_len \
       $lr $batch_size $epochs $weight_decay $max_norm
