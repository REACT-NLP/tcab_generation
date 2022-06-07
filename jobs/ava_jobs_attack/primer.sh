task_name=$1
dataset=$2
target_model_train_dataset=$3
model=$4
max_seq_len=$5
batch_size=$6
max_queries=$7
nodelist_=$8

toolchain_list=('textattack' 'openattack' 'None')

for toolchain in ${toolchain_list[@]}; do

    if [[ $toolchain = 'textattack' ]]; then
        attack_list=('bae' 'deepwordbug' 'faster_genetic' 'iga_wang' 'pruthi' 'pso' 'textbugger' 'textfooler' 'clean')
    elif [[ $toolchain = 'openattack' ]]; then
        attack_list=('genetic' 'hotflip' 'textbugger' 'viper')
    else
        attack_list=('clean')
    fi

    for attack in ${attack_list[@]}; do

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
               --gpus=1 \
               --cpus-per-task=4 \
               --job-name=$job_name \
               --output=jobs/logs/attack/$job_name \
               --error=jobs/errors/attack/$job_name \
               --account=ucinlp.a \
               jobs/ava_jobs_attack/runner.sh $task_name $dataset $target_model_train_dataset $model \
               $max_seq_len $batch_size $toolchain $attack $max_queries
    done
done
