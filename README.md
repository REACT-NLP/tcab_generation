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

Run the following script to generate adversarial against a target model trained on a select dataset with a select attack method from either the [TextAttack](https://github.com/QData/TextAttack) or [OpenAttack](https://github.com/thunlp/OpenAttack) toolchain.

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
After following the **Install** steps above, use the instructions below to extend TCAB with additional datasets or attacks.

### Adding a Domain Dataset
To add a new domain dataset:

1. Create a new directory in the `data` directory with the name of the dataset: `data/[dataset]/`.
2. Create a readme in the new directory describing exactly how to download the raw data, and how to preprocess it.
3. After preprocessing, there should be a `train/val/test.csv` files in that directory.

### Adding an Attack
To generate adversarial examples for a new attack, follow the steps in the **Attack** subsection under the **Dataset Generation** section above.

### Dataset and Target Models
You can find the TCAB dataset [here](https://zenodo.org/record/6615386#.YtLkUnbMKUl).
We also provide the target models used to generated these attacks on Google Drive. In case you need to download the models to a linux environment, consider using packages such as [gdrive](https://github.com/prasmussen/gdrive).
- [hatebase](https://drive.google.com/drive/folders/1j3siF5joTffwmCRTvchy2ELf6U6xpz6-)
- [wikipedia](https://drive.google.com/drive/folders/1jWdg2qZ6CG0LoLnGhg8i14GE-Xv7klnT)
- [civil comments](https://drive.google.com/drive/folders/1YP7FXtEA18hKBklyLYr6oa06avmrucSJ)
- [climate change waterloo](https://drive.google.com/drive/folders/19XCu3BmRjl80zEXSd5uMGtetZU9-LpeV)
- [imdb](https://drive.google.com/drive/folders/1sYzJCZAgZWuj8tfsx5ZRB-l8kbBcPadP)
- [sst-2](https://drive.google.com/drive/folders/1RrslogWv0djknDXxSU4PKRWQoFVWh-sJ)
