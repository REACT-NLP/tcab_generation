task_name=$1
dataset=$2
target_model_train_dataset=$3
model=$4
max_seq_len=$5
batch_size=$6
max_queries=$7
mem=$8
time=$9
partition=${10}
use_variants=${11}
constraint=""  # constraint should be empty string unless using preempt partition

if [[ $use_variants = 'use_variants' ]]; then
    toolchain_list=('textattack_variants')  # if 'use_variants', only use the attack variants
else
    toolchain_list=('textattack' 'openattack' 'none')
fi

for toolchain in ${toolchain_list[@]}; do

    if [[ $toolchain = 'textattack' ]]; then
        attack_list=('bae' 'bert' 'checklist' 'clare' 'deepwordbug' 'faster_genetic'
                     'genetic' 'hotflip' 'iga_wang' 'input_reduction'
                     'kuleshov' 'pruthi' 'pso' 'pwws' 'textbugger' 'textfooler')
    elif [[ $toolchain = 'openattack' ]]; then
        attack_list=('deepwordbug' 'fd' 'gan' 'genetic' 'hotflip' 'pso' 'pwws'
                     'textbugger' 'textfooler' 'uat' 'viper')
    elif [[ $toolchain = 'textattack_variants' ]]; then
#        attack_list=('deepwordbugv1' 'deepwordbugv2' 'deepwordbugv3'
#                     'pruthiv1' 'pruthiv2' 'pruthiv3'
#                     'textbuggerv1' 'textbuggerv2' 'textbuggerv3')
        attack_list=('deepwordbugv4' 'pruthiv4' 'textbuggerv4')
    else
        attack_list=('clean')
    fi

    for attack in ${attack_list[@]}; do

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
    done
done
