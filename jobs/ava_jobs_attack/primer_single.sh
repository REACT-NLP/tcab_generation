task_name=$1
dataset=$2
target_model_train_dataset=$3
model=$4
max_seq_len=$5
batch_size=$6
toolchain=$7
attack=$8
max_queries=$9
nodelist_=$10

if [[ $attack = 'pso' && $toolchain = 'openattack' ]];  then
    mem_=64
elif [[ $attack = 'bert' ]]; then
    mem_=128
else
    mem_=32
fi

job_name=A_${dataset}_${model}_${toolchain}_${attack}
sbatch --mem=${mem_}G \
       --time=4320 \
       --partition=ava_s.p \
       --nodelist=${nodelist_} \
       --cpus-per-task=4 \
       --gpus=1 \
       --job-name=$job_name \
       --output=jobs/logs/attack/$job_name \
       --error=jobs/errors/attack/$job_name \
       --account=ucinlp.a \
       jobs/ava_jobs_attack/runner.sh $task_name $dataset $target_model_train_dataset $model \
       $max_seq_len $batch_size $toolchain $attack $max_queries
