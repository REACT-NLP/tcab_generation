"""
Attacks a dataset given a trained model.
"""
import os
import sys
import time
import pickle
from datetime import datetime

# set environments
os.environ['TA_CACHE_DIR'] = '.cache'  # textattack cache
os.environ['TRANSFORMERS_CACHE'] = '.cache'  # transformers cache

import torch
import numpy as np
import pandas as pd
import evaluation as eval
from OpenAttack.utils import DataInstance
from textattack.attack_results import FailedAttackResult
from textattack.attack_results import SkippedAttackResult
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import utility
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # REACT root directory
from config.cmd_args import parse_classification_attack_args
from models import TextAttackDatasetWrapper
from models import TextAttackModelWrapper
from models import OpenAttackModelWrapper


def load_model(in_dir, model_name, device, max_seq_len=250, num_labels=2):
    """
    Loads trained model weights and sets model to evaluation mode.
    """

    if model_name in ['bert', 'roberta', 'xlnet']:
        model = utility.get_model(model_name=model_name, max_seq_len=max_seq_len, num_labels=num_labels)
        model.load_state_dict(torch.load(os.path.join(in_dir, 'weights.pt'), map_location=device))

    elif model_name == 'uclmr':

        # retrieve trained vectorizers
        tf_vectorizer = pickle.load(open(os.path.join(in_dir, 'tf_vectorizer.pkl'), 'rb'))
        tfidf_vectorizer = pickle.load(open(os.path.join(in_dir, 'tfidf_vectorizer.pkl'), 'rb'))

        # retrieve trained model
        model = utility.get_model(model_name=model_name,
                                  tf_vectorizer=tf_vectorizer,
                                  tfidf_vectorizer=tfidf_vectorizer)
        model.load_state_dict(torch.load(os.path.join(in_dir, 'weights.pt'), map_location=device))

    else:
        raise ValueError('unknown model_name {}'.format(model_name))

    # move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    return model


def predict(args, model, test_df, device, logger=None):
    """
    Generate predictions on a given test set.
    """
    start = time.time()

    if logger:
        logger.info('\nno. test samples: {:,}\n'.format(len(test_df)))

    # create dataloader
    test_data = test_df['text'].tolist()
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.model_batch_size)

    # generate predictions
    all_preds = []
    for batch, text_list in enumerate(test_dataloader):

        if logger:
            elpased = time.time() - start
            s = '[TEST] batch: {:,}, no. samples: {:,}...{:.3f}s'
            logger.info(s.format(batch + 1, args.model_batch_size * (batch + 1), elpased))

        # make predictions for this batch
        with torch.no_grad():
            preds = model(text_list)
            all_preds.append(preds.cpu().numpy().tolist())

    # concat all predictions
    all_preds = np.vstack(all_preds)
    y_pred = np.argmax(all_preds, axis=1)
    y_proba = softmax(all_preds, axis=1)

    # evaluate performance
    if logger:
        y_test = test_df['label'].values

        acc = accuracy_score(y_test, y_pred)
        s = '\nAcc.: {:.3f}'.format(acc)

        # multiclass classification
        if y_proba.shape[1] > 2:
            con_mat = confusion_matrix(y_test, y_pred)
            s += '\nConfusion matrix:\n{}'.format(con_mat)

        logger.info(s)

    return y_pred, y_proba


def save_results(args, results, out_dir, logger, done=False):
    """
    Save adversarial samples and summary.
    """

    # compile results
    df = pd.DataFrame(results)
    df['scenario'] = args.task_name
    df['attack_toolchain'] = args.attack_toolchain
    df['config_file'] = args.config_file0
    df['attacked_all_instances'] = done

    # rearrange columns
    all_cols = df.columns.tolist()
    required_cols = ['scenario',
                     'target_model_dataset', 'target_model',
                     'attack_toolchain', 'attack_name',
                     'original_text', 'perturbed_text',
                     'original_output', 'perturbed_output',
                     'status', 'config_file']
    extra_cols = [x for x in all_cols if x not in required_cols]
    df = df[required_cols + extra_cols]

    # save all samples
    pd.set_option('display.max_columns', 100)
    logger.info('\nresults:\n{}'.format(df))
    df.to_csv(os.path.join(out_dir, 'results.csv'), index=None)


def main(args):

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.random_seed)

    # make sure the dataset matches the scenario
    if args.dataset_name in ['nuclear_energy', 'climate-change_waterloo', 'imdb', 'sst']:
        assert args.task_name == 'sentiment'

    elif args.dataset_name in ['wikipedia', 'hatebase', 'civil_comments',
                               'wikipedia_personal', 'wikipedia_aggression',
                               'reddit_dataset', 'gab_dataset']:
        assert args.task_name == 'abuse'

    elif args.dataset_name == 'fnc1':
        assert args.task_name == 'fake_news'

    else:
        raise ValueError('unknown dataset_name {}'.format(args.dataset_name))

    # make sure there is no attack toolchain if generating 'clean' samples
    if args.attack_toolchain == 'none':
        assert args.attack_name == 'clean'

    elif args.attack_name == 'clean':
        assert args.attack_toolchain == 'none'

    # setup output directory
    out_dir = os.path.join(args.dir_out,
                           args.dataset_name,
                           args.model_name,
                           args.attack_toolchain,
                           args.attack_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = utility.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('timestamp: {}'.format(datetime.now()))

    # save config file
    utility.cmd_args_to_yaml(args, os.path.join(out_dir, 'config.yml'))

    # set no. labels
    num_labels = 2
    if args.dataset_name in ['nuclear_energy', 'climate-change_waterloo']:
        num_labels = 3

    # load trained model
    start = time.time()
    dir_model = os.path.join(args.dir_model, args.target_model_train_dataset, args.model_name)
    model = load_model(in_dir=dir_model,
                       model_name=args.model_name,
                       max_seq_len=args.model_max_seq_len,
                       num_labels=num_labels,
                       device=device)
    logger.info('loaded trained model...{:.3f}s'.format(time.time() - start))

    # read in the test set
    start = time.time()
    dir_dataset = os.path.join(args.dir_dataset, args.dataset_name)

    if os.path.exists(os.path.join(dir_dataset, 'test.csv')):
        test_df = pd.read_csv(os.path.join(dir_dataset, 'test.csv'))

    elif os.path.join(dir_dataset, 'data.csv'):
        test_df = pd.read_csv(os.path.join(dir_dataset, 'data.csv'))

    if 'index' in test_df:
        del test_df['index']

    assert test_df.columns.tolist() == ['text', 'label']
    test_df = test_df.reset_index().rename(columns={'index': 'test_index'})

    # encode labels for fake news
    if args.model_name == 'uclmr':
        label_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
        test_df['label'] = test_df['label'].apply(lambda x: label_map[x]).values

    # generate predictions on the test set
    pred_test, proba_test = predict(args, model, test_df, device, logger=logger)
    test_df['pred'] = pred_test
    logger.info('making predictions on the test set...{:.3f}s'.format(time.time() - start))

    # save predictions on the test set
    if args.attack_name == 'clean':
        test_text_list = test_df['text'].tolist()
        y_test = test_df['label'].values
        test_indices = test_df['test_index'].tolist()

        # record metadata for each prediction
        results = []
        for i, (pred, proba) in enumerate(zip(pred_test, proba_test)):
            text = test_text_list[i]
            proba = proba_test[i]

            result = {}
            result['target_model_train_dataset'] = args.target_model_train_dataset
            result['target_model_dataset'] = args.dataset_name
            result['target_model'] = args.model_name
            result['attack_name'] = args.attack_name
            result['test_index'] = test_indices[i]
            result['ground_truth'] = y_test[i]
            result['attack_time'] = 0
            result['status'] = 'clean'
            result['original_text'] = text
            result['original_output'] = proba
            result['perturbed_text'] = text
            result['perturbed_output'] = proba
            result['num_queries'] = 0
            result['frac_words_changed'] = 0
            results.append(result)

        save_results(args, results, out_dir, logger, done=True)

        # cleanup and exit
        utility.remove_logger(logger)
        exit(0)

    # get label names and encode labels if necessary
    if args.dataset_name in ['wikipedia', 'civil_comments', 'hatebase',
                             'wikipedia_personal', 'wikipedia_aggression',
                             'reddit_dataset', 'gab_dataset']:
        label_names = ['non-toxic', 'toxic']

    elif args.dataset_name == 'fnc1':
        label_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
        inverse_label_map = {v: k for k, v in label_map.items()}
        label_names = [inverse_label_map[i] for i in range(len(inverse_label_map))]

    elif args.dataset_name in ['imdb', 'sst']:
        label_names = ['negative', 'positive']

    elif args.dataset_name in ['nuclear_energy', 'climate-change_waterloo']:
        label_names = ['negative', 'neutral', 'positive']

    else:
        raise ValueError('unknown dataset {}'.format(args.dataset_name))

    # focus on correctly predicted TOXIC instances for abuse
    if args.task_name == 'abuse':
        temp_df = test_df[(test_df['label'] == 1) & (test_df['pred'] == 1)].copy()

    # focus on correctly predicted instances
    else:
        temp_df = test_df[test_df['label'] == test_df['pred']].copy()

    # attack text prioritizing longer text instances
    temp_df['length'] = temp_df['text'].apply(lambda x: len(x.split()))
    temp_df = temp_df.sort_values('length', ascending=False)
    temp_df = temp_df[:args.attack_n_samples].reset_index(drop=True)
    attack_indices = sorted(temp_df.index)
    logger.info('\nno. test: {:,}, no. candidates: {:,}'.format(len(test_df), len(attack_indices)))

    # result containers
    i = 0
    results = []
    begin = time.time()  # total time
    start = time.time()  # time per attack

    # selected indices information
    y_test = temp_df['label'].values
    test_indices = temp_df['test_index'].values

    # TextAttack
    if args.attack_toolchain in ['textattack', 'textattack_variants']:

        # prepare data
        dataset = TextAttackDatasetWrapper(list(zip(temp_df['text'], temp_df['label'])), label_names=label_names)

        # prepare attacker
        model_wrapper = TextAttackModelWrapper(model, batch_size=args.model_batch_size)
        attack_recipe = utility.get_attacker(args.attack_toolchain, args.attack_name)
        attack = attack_recipe.build(model_wrapper)
        attack.goal_function.query_budget = args.attack_max_queries

        # attack test set
        for attack_result in attack.attack_dataset(dataset, indices=attack_indices):
            end = time.time() - start
            cum_time = time.time() - begin

            # get attack status
            status = 'success'
            if isinstance(attack_result, FailedAttackResult):
                status = 'failed'
            elif isinstance(attack_result, SkippedAttackResult):
                status = 'skipped'

            # get original and peturbed objects
            og = attack_result.original_result
            pp = attack_result.perturbed_result

            num_words_changed = len(og.attacked_text.all_words_diff(pp.attacked_text))

            result = {}
            result['target_model_train_dataset'] = args.target_model_train_dataset
            result['target_model_dataset'] = args.dataset_name
            result['target_model'] = args.model_name
            result['attack_name'] = args.attack_name
            result['test_index'] = test_indices[attack_indices[i]]
            result['ground_truth'] = y_test[attack_indices[i]]
            result['attack_time'] = end
            result['status'] = status
            result['original_text'] = og.attacked_text.text
            result['original_output'] = og.raw_output.detach().cpu().tolist()
            result['perturbed_text'] = pp.attacked_text.text
            result['perturbed_output'] = pp.raw_output.detach().cpu().tolist()
            result['num_queries'] = attack_result.num_queries
            try:
                result['frac_words_changed'] = num_words_changed / len(og.attacked_text.words)
            except:
                result['frac_words_changed'] = -1
            results.append(result)

            s = 'Result {} (dataset={}, model={}, attack={}, no. queries={:,}, time={:.3f}s, cum. time={:.3f}s):'
            logger.info(s.format(i + 1, args.dataset_name, args.model_name, args.attack_name,
                                 result['num_queries'], end, cum_time))
            try:
                logger.info(attack_result.__str__(color_method='ansi'))
            except:
                logger.info(attack_result)
            logger.info('\n')

            # save results
            if (i + 1) % args.save_every == 0:
                logger.info('\nsaving results to {}/...'.format(out_dir))
                save_results(args, results, out_dir, logger)

            # update counters
            i += 1
            start = time.time()

        # save leftover results
        logger.info('\nsaving results to {}/...'.format(out_dir))
        save_results(args, results, out_dir, logger, done=True)

    # OpenAttack
    elif args.attack_toolchain == 'openattack':

        # prepare data
        dataset = [DataInstance(d) for d in temp_df['text'].tolist()]

        # add token separating head and body to "skip_words" for UCLMR model
        skip_words = ['|||||'] if args.model_name == 'uclmr' else None

        # prepare attacker
        # NOTE: add 'self.detach()' to PWWS and TextBugger .py files if numpy() RunTimeErrors occur
        model_wrapper = OpenAttackModelWrapper(model, attack_name=args.attack_name)
        attacker = utility.get_attacker(args.attack_toolchain, args.attack_name, skip_words=skip_words)
        attack_eval = eval.DefaultAttackEval(attacker, model_wrapper)

        # launch attacks and print attack results
        for result in attack_eval.eval(args, dataset, test_indices, y_test):
            results.append(result)

            if (i + 1) % args.save_every == 0:
                logger.info('\nsaving results to {}/...'.format(out_dir))
                save_results(args, results, out_dir, logger)

            i += 1

        # save results
        logger.info('\nsaving results to {}/...'.format(out_dir))
        save_results(args, results, out_dir, logger, done=True)

    else:
        raise ValueError('unknown toolchain {}'.format(args.attack_toolchain))

    logger.info('\ntotal time: {:.3f}s'.format(time.time() - begin))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--config_file0', is_config_file=True, help='YAML config file')
    parser.add_argument('--dir_model', type=str, default='target_models/', help='central directory for trained models.')
    parser.add_argument('--dir_dataset', type=str, default='data/', help='central directory for storing datasets.')
    parser.add_argument('--dir_out', type=str, default='attacks/', help='central directory to store attacks.')

    # Experiment settings
    parser.add_argument('--task_name', type=str, help='e.g., abuse, sentiment, summarization')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='roberta', help='Model type.')
    parser.add_argument('--model_max_seq_len', type=int, default=250, help='Max. no. tokens per string.')
    parser.add_argument('--model_batch_size', type=int, default=32, help='No. instances per mini-batch.')

    # Data parameters
    parser.add_argument('--dataset_name', type=str, help='Dataset to attack.')
    parser.add_argument('--target_model_train_dataset', type=str, help='Dataset used to train the target model.')

    # Attack parameters
    parser.add_argument('--attack_toolchain', type=str, default='textattack', help='e.g., textattack or openattack')
    parser.add_argument('--attack_name', type=str, default='bae', help='Name of the attack.')
    parser.add_argument('--attack_max_queries', type=int, default=500, help='Max. no. queries per attack')
    parser.add_argument('--attack_n_samples', type=int, default=1000000000, help='No. samples to attack; if 0, attack all samples')

    # Other parameters
    parser.add_argument('--random_seed', type=int, default=1, help='the random seed value to use for reproducibility')
    parser.add_argument('--save_every', type=int, default=25, help='no. samples to attack between saving.')

    args = parser.parse_args()
    main(args)
