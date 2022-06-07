"""
Organize results into two CSVs.
"""
import os
import string
import argparse
from datetime import datetime
from itertools import product

import pandas as pd
from tqdm import tqdm

import utility


def remove_punctuation(s):
    """
    Remove all punctuation from the string s.
    """
    return s.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(s):
    """
    Remove all numbers from the string s.
    """
    return ''.join(c for c in s if not c.isdigit())


def get_word_count(s):
    """
    Compute no. words in the string after removing punctuation and numbers.
    """
    return len(remove_numbers(remove_punctuation(s)).split())


def filter_non_sucesses(attack_df):
    """
    Filter out unsucessful adversarial attacks.
    """
    df = attack_df.copy()
    temp_df1 = df[df['status'] == 'success']
    temp_df2 = df[df['attack_name'] == 'original']
    result_df = pd.concat([temp_df1, temp_df2])
    return result_df


def filter_short_strings(attack_df, min_num_words=10):
    """
    Filter out strings that do not meet the min. no. words.
    """
    df = attack_df.copy()
    df['n_words'] = df['original_text'].apply(lambda x: get_word_count(x))
    df = df[df['n_words'] >= min_num_words]
    del df['n_words']
    return df


def create_unique_ids(attack_df):
    """
    Create a unique ID for each attack instance.
    """
    df = attack_df.copy()
    df = df.reset_index().reset_index()
    df = df.rename(columns={'level_0': 'attack_id'})
    del df['index']
    return df


def extract_summary_statistics(attacks_df):
    """
    Compute statistics about the attacks
    for each dataset/attack/model combination.
    """
    results = []
    group = ['target_model_dataset', 'target_model', 'attack_toolchain', 'attack_name']
    for (target_model_dataset, target_model, attack_toolchain, attack_name), df in attacks_df.groupby(group):
        df['num_words'] = df['original_text'].apply(lambda x: get_word_count(x))
        qf = df[df['status'] == 'success']

        # compute statistics
        num_attacks = len(df)
        num_success_attacks = len(qf)
        num_failed_attacks = len(df[df['status'] == 'failed'])
        num_skipped_attacks = len(df[df['status'] == 'skipped'])
        pct_nonskipped_attacks = (num_attacks - num_skipped_attacks) / num_attacks
        pct_failed_attacks = num_failed_attacks / num_attacks
        attack_success_rate = 0
        if num_success_attacks + num_failed_attacks != 0:
            attack_success_rate = num_success_attacks / (num_success_attacks + num_failed_attacks)

        # save statistics
        result = {}
        result['target_model_dataset'] = target_model_dataset
        result['target_model'] = target_model
        result['attack_toolchain'] = attack_toolchain
        result['attack_name'] = attack_name
        result['num_attacks'] = num_attacks
        result['num_success'] = num_success_attacks
        result['num_failed'] = num_failed_attacks
        result['num_skipped'] = num_skipped_attacks
        result['pct_nonskipped_attacks'] = pct_nonskipped_attacks
        result['pct_failed_attacks'] = pct_failed_attacks
        result['attack_success_rate'] = attack_success_rate
        result['avg_frac_words_changed'] = qf['frac_words_changed'].mean()
        result['avg_num_input_words'] = qf['num_words'].mean()
        result['avg_num_queries'] = qf['num_queries'].mean()
        results.append(result)

    summary_df = pd.DataFrame(results)
    return summary_df


def get_result(in_dir):
    """
    Obtain results.
    """
    sample_fp = os.path.join(in_dir, 'results.csv')

    # load sample results
    if os.path.exists(sample_fp):
        sample_df = pd.read_csv(sample_fp)

    else:
        sample_df = None

    return sample_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    # prepraring experiment combinations
    experiment_settings = list(product(*[args.target_model_dataset,
                                         args.target_model,
                                         args.attack_toolchain,
                                         args.attack_name]))

    # results container
    results = []

    # retrieve results from each experiment
    for target_model_dataset, target_model, attack_toolchain, attack_name in tqdm(experiment_settings):

        # get attack directory
        experiment_dir = os.path.join(args.in_dir,
                                      target_model_dataset,
                                      target_model,
                                      attack_toolchain,
                                      attack_name)

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        # add results to result dicts
        result = get_result(experiment_dir)

        if result is not None:
            results.append(result)

    # set Pandas options
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)

    # combine results from different experiments
    attack_df = pd.concat(results)
    logger.info('\nAttack results:\n{}'.format(attack_df))

    # compute attack statistics
    summary_df = extract_summary_statistics(attack_df)
    logger.info('\nSummary results:\n{}'.format(summary_df))

    # filter attack results
    logger.info('\nfiltering out unsuccessful and short samples...')
    attack_df = filter_non_sucesses(attack_df)
    attack_df = filter_short_strings(attack_df, min_num_words=args.min_num_words)
    attack_df = create_unique_ids(attack_df)
    logger.info('\nFiltered attack results:\n{}'.format(attack_df))
    logger.info('\nAttack count:\n{}\n'.format(attack_df['attack_name'].value_counts().to_dict()))

    # save aggregated results
    logger.info('\nSaving {}...'.format(os.path.join(out_dir, 'attack.csv')))
    attack_df.to_csv(os.path.join(out_dir, 'attack_dataset.csv'), index=None)

    logger.info('Saving {}...'.format(os.path.join(out_dir, 'attack_summary.csv')))
    summary_df.to_csv(os.path.join(out_dir, 'attack_summary.csv'), index=None)


def main(args):

    out_dir = os.path.join(args.out_dir)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = utility.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    create_csv(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='attacks/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='attacks/processed_results/', help='output directory.')

    # Experiment settings
    parser.add_argument('--target_model_dataset', type=str, nargs='+',
                        default=['nuclear_energy',
                                 'climate-change_waterloo',
                                 'imdb',
                                 'sst',
                                 'wikipedia',
                                 'hatebase',
                                 'civil_comments',
                                 'fnc1'], help='dataset.')
    parser.add_argument('--target_model', type=str, nargs='+', help='model.',
                        default=['bert',
                                 'roberta',
                                 'xlnet',
                                 'uclmr'])
    parser.add_argument('--attack_toolchain', type=str, nargs='+', help='toolchain.',
                        default=['textattack',
                                 'openattack'])
    parser.add_argument('--attack_name', type=str, nargs='+', help='method.',
                        default=['original',
                                 'bae',
                                 'bert',
                                 'checklist',
                                 'clare',
                                 'deepwordbug',
                                 'faster_genetic',
                                 'fd',
                                 'gan',
                                 'genetic',
                                 'hotflip',
                                 'iga_wang',
                                 'input_reduction',
                                 'kuleshov',
                                 'pruthi',
                                 'pso',
                                 'pwws',
                                 'textbugger',
                                 'textfooler',
                                 'uat',
                                 'viper'])

    # Filter settings
    parser.add_argument('--min_num_words', type=int, default=10, help='required min. no. words per text.')

    args = parser.parse_args()
    main(args)
