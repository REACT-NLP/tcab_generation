task_name=$1
dataset=$2
target_model_train_dataset=$3
model=$4
max_seq_len=$5
batch_size=$6
toolchain=$7
attack=$8
max_queries=$9
mem=${10}
time=${11}
partition=${12}
constraint=""  # constraint should be empty string unless using preempt partition

if [[ $attack = 'pso' && $toolchain = 'openattack' ]];  then
    partition_=preempt
    constraint_=volta
    mem_=64
    time_=$time
elif [[ $attack = 'bert' ]]; then
    partition_=$partition
    constraint_=$constraint
    mem_=128
    time_=$time
else
    partition_=$partition
    constraint_=$constraint
    mem_=$mem
    time_=$time
fi

job_name=A_${dataset}_${model}_${toolchain}_${attack}
sbatch --mem=${mem_}G \
       --time=$time_ \
       --partition=$partition_ \
       --constraint=$constraint_ \
       --gres=gpu:1 \
       --job-name=$job_name \
       --output=jobs/logs/attack/$job_name \
       --error=jobs/errors/attack/$job_name \
       jobs/attack/runner.sh $task_name $dataset $target_model_train_dataset $model \
       $max_seq_len $batch_size $toolchain $attack $max_queries
