"""
Train/fine-tune a text classification model.
"""
import os
import time
import pickle
import argparse
from datetime import datetime

# set environments
os.environ['TA_CACHE_DIR'] = '.cache'  # TextAttack cache
os.environ['TRANSFORMERS_CACHE'] = '.cache'  # transformers cache

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from scipy.special import softmax

import utility
from models import stop_words


def train(args, model, train_dataloader, optimizer, loss_fn, device, max_norm=1.0, logger=None):
    """
    Execute one epoch of training using mini-batches.
    """
    start = time.time()

    # activates dropout, allows gradients
    model.train()

    # iterate over batches
    train_loss = 0
    for step, (text_list, labels) in enumerate(train_dataloader):

        if logger:
            elapsed = time.time() - start
            s = '[TRAIN] batch {:,}, no. samples: {:,}...{:.3f}s'
            logger.info(s.format(step + 1, args.batch_size * (step + 1), elapsed))

        # push the batch to device
        labels = labels.to(device)

        optimizer.zero_grad()  # clear previous gradients

        # get predictions for this batch
        preds = model(text_list)

        loss = loss_fn(preds, labels)  # compute loss
        train_loss += loss.item()  # add to total loss

        loss.backward()  # calculate gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # exploding gradient
        optimizer.step()  # update parameters

    # compute loss during this epoch
    avg_loss = train_loss / len(train_dataloader)

    return avg_loss


def evaluate(args, model, val_dataloader, loss_fn, device, logger=None):
    """
    Test model performance for this epoch on a validation set.
    """
    start = time.time()

    # disable dropout for deterministic output
    model.eval()

    # batch prediction
    val_loss = 0
    for step, (text_list, labels) in enumerate(val_dataloader):

        if logger:
            elapsed = time.time() - start
            s = '[EVAL] batch {:,}, no. samples: {:,}...{:.3f}s'
            logger.info(s.format(step + 1, args.batch_size * (step + 1), elapsed))

        # push the batch to device
        labels = labels.to(device)

        # deactivate autograd, reduces memory and speeds up computation
        with torch.no_grad():

            # get predictions for this batch
            preds = model(text_list)

            loss = loss_fn(preds, labels)  # compute loss
            val_loss += loss.item()  # add to total loss

    val_loss /= len(val_dataloader)

    return val_loss


def test(args, model, test_dataloader, y_test, device, logger=None):
    """
    Evaluate performance of the model on a
    held-out test set.
    """
    start = time.time()

    # result container
    result = {}

    # activate evaluation mode
    model.eval()

    # generate predictions on the test set
    all_preds = []
    for step, (text_list, labels) in enumerate(test_dataloader):

        if logger:
            elapsed = time.time() - start
            s = '[TEST] batch {:,}, no. samples: {:,}...{:.3f}s'
            logger.info(s.format(step + 1, args.batch_size * (step + 1), elapsed))

        # make predictions for this batch
        with torch.no_grad():
            preds = model(text_list)
            all_preds.append(preds.cpu().numpy().tolist())

    # concat all predictions
    all_preds = np.vstack(all_preds)
    y_pred = np.argmax(all_preds, axis=1)
    y_proba = softmax(all_preds, axis=1)

    # compute scores
    result['acc'] = accuracy_score(y_test, y_pred)

    # extra metrics for binary classification models
    if y_proba.shape[1] == 2:
        result['auc'] = roc_auc_score(y_test, y_proba[:, 1])
        result['ap'] = average_precision_score(y_test, y_proba[:, 1])
        result['f1'] = f1_score(y_test, y_pred)
    else:
        result['auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
        result['ap'] = -1
        result['f1'] = f1_score(y_test, y_pred, average='macro')

    # save predictions
    result['pred'] = y_pred
    result['proba'] = y_proba

    return result


def main(args):

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.rs)

    # setup output directory
    out_dir = os.path.join(args.out_dir, args.dataset, args.model)
    os.makedirs(out_dir, exist_ok=True)
    logger = utility.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('timestamp: {}'.format(datetime.now()))

    # read in data
    start = time.time()
    train_df = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'train.csv'), nrows=args.nrows)
    val_df = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'val.csv'), nrows=args.nrows)
    test_df = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'test.csv'), nrows=args.nrows)
    logger.info('\nreading in data...{:.3f}s'.format(time.time() - start))

    # shuffle data
    start = time.time()
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    logger.info('shuffling training data...{:.3f}s'.format(time.time() - start))

    # basic dataset stats
    logger.info('\nno. train: {:,}'.format(len(train_df)))
    logger.info('no. val: {:,}'.format(len(val_df)))
    logger.info('no. test: {:,}\n'.format(len(test_df)))

    # UCLMR-specific configurations
    if args.model == 'uclmr':
        gold_df = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'test_competition.csv'))
        train_df = pd.concat([train_df, val_df])
        val_df = train_df.copy()

        # encode labels
        label_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
        inverse_label_map = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
        train_df['label'] = train_df['label'].apply(lambda x: label_map[x])
        val_df['label'] = val_df['label'].apply(lambda x: label_map[x])
        test_df['label'] = test_df['label'].apply(lambda x: label_map[x])

        # frequency counter
        start = time.time()
        tf_vectorizer = TfidfVectorizer(use_idf=False, stop_words=stop_words, max_features=5000)
        tf_vectorizer.fit(train_df['text'].tolist())
        logger.info('\nfitting tf_vectorizer...{:.3f}s'.format(time.time() - start))

        # tfidf transformer
        start = time.time()
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
        tfidf_vectorizer.fit(train_df['text'].tolist() + test_df['text'].tolist())
        logger.info('fitting tfidf_vectorizer...{:.3f}s'.format(time.time() - start))

        # save vectorizers
        start = time.time()
        pickle.dump(tf_vectorizer, open(os.path.join(out_dir, 'tf_vectorizer.pkl'), 'wb'))
        pickle.dump(tfidf_vectorizer, open(os.path.join(out_dir, 'tfidf_vectorizer.pkl'), 'wb'))
        logger.info('saving tf_vectorizer and tfidf_vectorizer...{:.3f}s'.format(time.time() - start))

        # get model
        start = time.time()
        model = utility.get_model(model_name=args.model,
                                  tf_vectorizer=tf_vectorizer,
                                  tfidf_vectorizer=tfidf_vectorizer)
        model = model.to(device)
        logger.info('\nloading the model...{:.3f}s'.format(time.time() - start))

    # BERT, RoBERTa, and XLNET models
    else:
        num_classes = len(train_df['label'].unique())
        start = time.time()
        model = utility.get_model(model_name=args.model, max_seq_len=args.max_seq_len, num_labels=num_classes)
        model = model.to(device)
        logger.info('\nloading the model...{:.3f}s'.format(time.time() - start))

    # create dataloaders
    train_data = list(zip(train_df['text'].tolist(), np.array(train_df['label'].values)))
    val_data = list(zip(val_df['text'].tolist(), np.array(val_df['label'].values)))
    test_data = list(zip(test_df['text'].tolist(), np.array(test_df['label'].values)))

    train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.batch_size)
    logger.info('creating dataloaders...{:.3f}s'.format(time.time() - start))

    # define loss function and optimizer
    weight_decay = args.weight_decay if args.model == 'uclmr' else 0.0
    loss_fn = utility.get_loss_fn(args.loss_fn)
    optimizer = utility.get_optimizer(args.optimizer, args.lr, model, weight_decay=weight_decay)

    # set initial loss to infinite
    best_val_loss = float('inf')

    # train model on multiple epochs
    logger.info('\nTraining')
    begin = time.time()

    # training loop
    for epoch in range(args.epochs):
        start = time.time()

        # train and evaluate
        train_loss = train(args, model, train_dataloader, optimizer, loss_fn, device, args.max_norm, logger=logger)
        val_loss = evaluate(args, model, val_dataloader, loss_fn, device, logger=logger)

        # progress update
        end = time.time() - start
        s = '[{}, Epoch {}] train loss: {:.3f}, val loss: {:.3f}, time: {:.3f}s'
        logger.info(s.format(args.model, epoch + 1, train_loss, val_loss, end))

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info('  saving model...')
            torch.save(model.state_dict(), os.path.join(out_dir, 'weights.pt'))

            # test model
            y_test = test_df['label'].values
            res = test(args, model, test_dataloader, y_test, device, logger=logger)
            s = '[Test] Acc.: {:.3f}, AUC: {:.3f}, AP: {:.3f}, F1: {:.3f}'
            logger.info(s.format(res['acc'], res['auc'], res['ap'], res['f1']))

            # save results
            results = {}
            results['dataset'] = args.dataset
            results['model'] = args.model
            results['acc'] = res['acc']
            results['auc'] = res['auc']
            results['ap'] = res['ap']
            results['f1'] = res['f1']
            results['train_time'] = time.time() - begin
            results['max_seq_len'] = args.max_seq_len
            results['loss_fn'] = args.loss_fn
            results['lr'] = args.lr
            results['optimizer'] = args.optimizer
            results['max_norm'] = args.max_norm
            results['batch_size'] = args.batch_size
            results['epochs'] = args.epochs
            np.save(os.path.join(out_dir, 'results.npy'), results)

            # save predictions
            if args.model == 'uclmr':
                columns = ['Headline', 'Body ID', 'Stance']
                submission_df = gold_df.copy()
                submission_df['Stance'] = res['pred']
                submission_df['Stance'] = submission_df['Stance'].apply(lambda x: inverse_label_map[x].lower())
                submission_df.to_csv(os.path.join(out_dir, 'predictions.csv'), columns=columns, index=None)
                test_time = time.time() - start
                logger.info('[TEST] time: {:.3f}s\n{}'.format(test_time, submission_df['Stance'].value_counts()))

    # clean up
    utility.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--out_dir', type=str, default='target_models/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')

    # experiment settings
    parser.add_argument('--dataset', default='wikipedia', help='dataset to use for the experiment.')
    parser.add_argument('--model', type=str, default='roberta', help='model type.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')

    # training parameters
    parser.add_argument('--loss_fn', type=str, default='crossentropy', help='loss function to optimize over.')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimization strategy.')
    parser.add_argument('--max_seq_len', type=int, default=250, help='maximum number of tokens per string.')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='number of sentences per mini-batch.')
    parser.add_argument('--epochs', type=int, default=10, help='number of full training rounds.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='l2 regularization.')
    parser.add_argument('--max_norm', type=float, default=1.0, help='max. norm for gradient clipping.')

    # additional settings
    parser.add_argument('--nrows', type=int, default=1000000000, help='number of train and test samples.')

    args = parser.parse_args()
    main(args)
