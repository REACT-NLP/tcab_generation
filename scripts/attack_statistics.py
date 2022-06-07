"""
Script that summarizes attack success rates.
"""
import os
import argparse

import pandas as pd

import utility


def main(args):

    os.makedirs(args.log_dir, exist_ok=True)
    logger = utility.get_logger(os.path.join(args.log_dir, 'attack_statistics.txt'))
    
    # find attack results
    for root, dirs, files in os.walk(args.attack_dir):

        for file in files:
            if file == 'results.csv':
                df = pd.read_csv(os.path.join(root, file))

                tm_dataset = df.iloc[0]['target_model_dataset']
                tm = df.iloc[0]['target_model']
                attack = df.iloc[0]['attack_name']
                toolchain = df.iloc[0]['attack_toolchain']

                logger.info(f'[TM_dataset: {tm_dataset:>10}, TM: {tm:>10}, attack: {attack:>15}, '
                            f'toolchain: {toolchain:>10}]'
                            f' No. success: {len(df[df["status"] == "success"]):>10,}/{len(df):>10,}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_dir', type=str, default='attacks')
    parser.add_argument('--log_dir', type=str, default='analysis')
    args = parser.parse_args()
    main(args)
