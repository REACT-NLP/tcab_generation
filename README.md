TCAB Dataset Generation
===

This pipeline describes the process for generating and extending the text classification attack benchmark (TCAB) dataset below.

## Install
1. Install Python 3.8+.
1. Install Python packages: `pip3 install -r requirements.txt`.

## Dataset Generation

Follow the instructions below to generate perturbations for a select domain dataset.

### Domain Data
Download and preprocess a domain dataset:

1. Change directories to a select dataset: `cd data/[dataset]/`.
2. Follow the readme in that directory for downloading and preprocessing.

### Train

Run the following script to train a target model on a select dataset.

`python3 scripts/train.py` with arguments:

* `--dataset`: `wikipedia`, `civil_comments`, `hatebase`, `imdb`, `climate-change_waterloo`, `sst`.
* `--model`: `bert`, `roberta`, `xlnet`.
* `--loss_fn`: Loss function used during training (default: `crossentropy`).
* `--optimizer`: Optimizer used to optimize the loss function (default: `adam`).
* `--max_seq_len`: Max. no. tokens fed into the model at a time (default: `250`).
* `--lr`: learning rate (default: `1e-6`).
* `--batch_size`: No. samples used for mini-batches (default: `32`).
* `--epochs`: No. times through the entire training set (default: `10`).
* `--weight_decay`: Weight decay used in the `adam` optimizer (default: `0.0`).
* `--max_norm`: Controls exploding gradients (default: `1.0`).

The learned target model will be saved to `target_models/[dataset]/[model]/`.

### Attack

Run the following script to generate adversarial against a target model trained on a select dataset with a select attack method from either the TextAttack or OpenAttack toolchain.

`python3 scripts/attack.py` with arguments:

* `--task_name`: `sentiment`, `abuse`.
* `--dataset_name`: `wikipedia`, `civil_comments`, `hatebase`, `imdb`, `climate-change_waterloo`, `sst`.
* `--target_model_train_dataset`: `wikipedia`, `civil_comments`, `hatebase`, `imdb`, `climate-change_waterloo`, `sst`.
* `--model_name`: `bert`, `roberta`, `xlnet` (default: `roberta`).
* `--model_max_seq_len`: Max. no. tokens fed into the model at a time (default: `250`).
* `--model_batch_size`: No. samples used for mini-batches (default: `32`).
* `--attack_toolchain`: `textattack` or `openattack` (default: `textattack`).
* `--attack_name`: `bae`, `bert`, `checklist`, `clare`,  `deepwordbug`, `faster_genetic`, `fd`, `gan`, `genetic`, `hotflip`, `iga_wang`, `input_reduction`, `kuleshov`, `pruthi`, `pso`, `pwws`, `textbugger`, `textfooler` `uat`, or `viper` (default: `bae`).
* `--attack_max_queries`: Max. no. queries per attack (default: `500`).

Results are saved into `attacks/[dataset_name]/[model_name]/[attack_toolchain]/[attack]/` and includes a CSV with the following columns:

* `target_model_dataset`: Dataset being attacked.
* `target_model_train_dataset`: Dataset used to train the model being attacked.
* `target_model`: Name of the model being attacked.
* `attack_name`: Name of the attack used to perturb the input.
* `test_index`: Unique index of the test instance with respect to the `target_model_dataset`.
* `attack_time`: Time taken per attack.
* `ground_truth`: Actual label of the test instance.
* `status`: `success`, `failure`, or `skipped` (`textattack` only).
* `original_text`: Original input text.
* `original_output`: Original output distribution.
* `perturbed_text`: Post-perturbation text.
* `perturbed_output`: Post-perturbation output distribution.
* `num_queries`: No. queries used during the attack (`textattack` only).
* `frac_words_changed`: Fraction of words changed in a successful attack.


## Extending TCAB
