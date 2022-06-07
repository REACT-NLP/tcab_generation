"""
Utility methods to make life easier.
"""
import os
import sys
import time
import shutil
import logging
import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from filelock import FileLock
from filelock import Timeout
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textattack
import OpenAttack

import attack_variants
from models import BERTClassifier
from models import RoBERTaClassifier
from models import XLNetClassifier
from models import UCLMRClassifier


def cmd_args_to_yaml(cmd_args, outfile_name, ignore_list=[]):
    """
    Takes cmd_args, an argparse.Namespace object, and writes the values to a file
    in YAML format. Some parameter values might not need to be saved, so you can
    pass a list of parameter names as the ignore_list, and the values for these
    parameter names will not be saved to the YAML file.
    """
    cmd_args_dict = vars(cmd_args)
    with open(outfile_name, 'w') as yaml_outfile:
        for parameter, value in cmd_args_dict.items():
            # don't write the parameter value if parameter in the
            # ignore list or the value of the parameter is None
            if parameter in ignore_list or value is None:
                continue
            else:
                # write boolean values as 1's and 0's
                if isinstance(value, bool):
                    value = int(value)
                yaml_outfile.write(f'{parameter}: {value}\n')


class Cache:
    """
    Cache that holds extracted feature values.
    """
    def __init__(self, dir_cache, fn):
        self.dir_cache = dir_cache
        self.fn = fn

        self.fp_ = os.path.join(self.dir_cache, self.fn)
        self.lock_fp_ = '{}.lock'.format(self.fp_)

        # check cache exists
        if os.path.exists(self.fp_):
            lock = FileLock(self.lock_fp_, timeout=100)

            try_count = 0
            while try_count < 100:

                try:

                    # acquire lock on file and read in cache, automatically gets released
                    with lock.acquire(timeout=100):  # give it 100 seconds to acquire lock
                        self.cache_ = np.load(self.fp_, allow_pickle=True)[()]

                # could not read cache file, trying again in 1 second...
                except:
                    time.sleep(1)
                    try_count += 1

                # success
                else:
                    break

        # create new cache
        else:
            os.makedirs(self.dir_cache, exist_ok=True)
            self.cache_ = {}

    def load_vals(self, feature_set, unique_ids):
        """
        Retrieves values for the specified feature set for
        each unique instance.

        Returns a 2d numpy array of shape=(len(unique_ids), no. feature values), and
                a 1d list of feature names
        """
        unique_ids = self._convert_to_list(unique_ids)
        assert feature_set in self.cache_, '{} not in cache!'.format(feature_set)

        # get feature values and names
        vals_dict = self.cache_[feature_set]['values']
        feature_names = self.cache_[feature_set]['names']

        # collect values
        feature_vals = np.vstack([vals_dict[unique_id] for unique_id in unique_ids])

        # make sure feature names align with the no. feature values
        assert feature_vals.shape[1] == len(feature_names)

        return feature_vals, feature_names

    def store_vals(self, feature_set, feature_vals, feature_names, unique_ids):
        """
        Stores the feature values for each instance and the feature names
        in the cache.
        """
        unique_ids = self._convert_to_list(unique_ids)
        assert feature_vals.shape[0] == len(unique_ids), 'no. instances do not match!'
        assert feature_vals.shape[1] == len(feature_names), 'no. features do not match!'

        # create dict for this feature set if it does not already exist
        if feature_set not in self.cache_:
            self.cache_[feature_set] = {'names': feature_names, 'values': {}}
            self.cache_[feature_set]['names'] = feature_names
            vals_dict = self.cache_[feature_set]['values']

        # dict already exists
        else:
            assert len(feature_names) == len(self.cache_[feature_set]['names']), 'feature names do not match!'
            vals_dict = self.cache_[feature_set]['values']

        # add/update instances in feature values dictionary
        for i in range(feature_vals.shape[0]):
            vals_dict[unique_ids[i]] = feature_vals[i]

    def saved_ids(self, feature_set, unique_ids):
        """
        Returns set of saved ids in `unique_ids`.
        """
        result = set()

        # get sved ids
        if feature_set in self.cache_:
            vals_dict = self.cache_[feature_set]['values']
            result = set(unique_ids).intersection(set(vals_dict.keys()))

        return result

    def save(self):
        """
        Saves current state of the cache to the initialized filepath.
        """

        # create lock object
        lock = FileLock(self.lock_fp_, timeout=100)

        try:

            # acquire lock on file and save cache, automatically gets released
            with lock.acquire(timeout=100):  # give it 100 seconds to acquire lock
                np.save(self.fp_, self.cache_)

        except Timeout:
            raise IOError('Could not acquire lock on cache.')

    def _convert_to_list(self, unique_ids):
        """
        Convert to list.
        """
        if type(unique_ids) != list:
            unique_ids = list(unique_ids)
        return unique_ids


def get_logger(filename=''):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.handlers = []  # clear previous handlers
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def remove_logger(logger):
    """
    Remove handlers from logger.
    """
    logger.handlers = []


def clear_dir(in_dir):
    """
    Clear contents of directory.
    """
    if not os.path.exists(in_dir):
        return -1

    # remove contents of the directory
    for fn in os.listdir(in_dir):
        fp = os.path.join(in_dir, fn)

        # directory
        if os.path.isdir(fp):
            shutil.rmtree(fp)

        # file
        else:
            os.remove(fp)

    return 0


def get_model(model_name, max_seq_len=250, num_labels=2, tf_vectorizer=None, tfidf_vectorizer=None):
    """
    Return a new instance of the text classification model.
    """
    if model_name == 'bert':
        model = BERTClassifier(max_seq_len=max_seq_len, num_labels=num_labels)

    elif model_name == 'roberta':
        model = RoBERTaClassifier(max_seq_len=max_seq_len, num_labels=num_labels)

    elif model_name == 'xlnet':
        model = XLNetClassifier(max_seq_len=max_seq_len, num_labels=num_labels)

    elif model_name == 'uclmr':
        assert tf_vectorizer is not None
        assert tfidf_vectorizer is not None
        model = UCLMRClassifier(tf_vectorizer=tf_vectorizer, tfidf_vectorizer=tfidf_vectorizer)

    else:
        raise ValueError('Unknown model {}!'.format(model_name))

    return model


def get_loss_fn(loss_fn_name, weight=None):
    """
    Choose loss function.
    """
    if loss_fn_name == 'crossentropy':
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    else:
        raise ValueError('Unknown loss_fn {}'.format(loss_fn_name))

    return loss_fn


def get_optimizer(optimizer_name, lr, model, weight_decay=0.0):
    """
    Choose optimizer.
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    else:
        raise ValueError('Unknown optimizer {}'.format(optimizer_name))

    return optimizer


def get_attacker(attack_toolchain, attack_name, skip_words=None):
    """
    Load attacker.
    """

    if attack_toolchain in ['textattack', 'textattack_variants']:

        if attack_name == 'bae':
            attacker = textattack.attack_recipes.BAEGarg2019

        elif attack_name == 'bert':
            attacker = textattack.attack_recipes.BERTAttackLi2020

        elif attack_name == 'checklist':
            attacker = textattack.attack_recipes.CheckList2020

        elif attack_name == 'clare':
            attacker = textattack.attack_recipes.CLARE2020

        elif attack_name == 'deepwordbug':
            attacker = textattack.attack_recipes.DeepWordBugGao2018

        elif attack_name == 'deepwordbugv1':
            attacker = attack_variants.DeepWordBugGao2018V1

        elif attack_name == 'deepwordbugv2':
            attacker = attack_variants.DeepWordBugGao2018V2

        elif attack_name == 'deepwordbugv3':
            attacker = attack_variants.DeepWordBugGao2018V3

        elif attack_name == 'deepwordbugv4':
            attacker = attack_variants.DeepWordBugGao2018V4

        elif attack_name == 'faster_genetic':
            attacker = textattack.attack_recipes.FasterGeneticAlgorithmJia2019

        elif attack_name == 'genetic':
            attacker = textattack.attack_recipes.GeneticAlgorithmAlzantot2018

        elif attack_name == 'hotflip':
            attacker = textattack.attack_recipes.HotFlipEbrahimi2017

        elif attack_name == 'iga_wang':
            attacker = textattack.attack_recipes.IGAWang2019

        elif attack_name == 'input_reduction':
            attacker = textattack.attack_recipes.InputReductionFeng2018

        elif attack_name == 'kuleshov':
            attacker = textattack.attack_recipes.Kuleshov2017

        elif attack_name == 'pruthi':
            attacker = textattack.attack_recipes.Pruthi2019

        elif attack_name == 'pruthiv1':
            attacker = attack_variants.Pruthi2019V1

        elif attack_name == 'pruthiv2':
            attacker = attack_variants.Pruthi2019V2

        elif attack_name == 'pruthiv3':
            attacker = attack_variants.Pruthi2019V3

        elif attack_name == 'pruthiv4':
            attacker = attack_variants.Pruthi2019V4

        elif attack_name == 'pso':
            attacker = textattack.attack_recipes.PSOZang2020

        elif attack_name == 'pwws':
            attacker = textattack.attack_recipes.PWWSRen2019

        elif attack_name == 'textbugger':
            attacker = textattack.attack_recipes.TextBuggerLi2018

        elif attack_name == 'textbuggerv1':
            attacker = attack_variants.TextBuggerLi2018V1

        elif attack_name == 'textbuggerv2':
            attacker = attack_variants.TextBuggerLi2018V2

        elif attack_name == 'textbuggerv3':
            attacker = attack_variants.TextBuggerLi2018V3

        elif attack_name == 'textbuggerv4':
            attacker = attack_variants.TextBuggerLi2018V4

        elif attack_name == 'textfooler':
            attacker = textattack.attack_recipes.TextFoolerJin2019

        else:
            raise ValueError('unknown attack {}'.format(attack_name))

    elif attack_toolchain == 'openattack':

        if attack_name == 'deepwordbug':
            attacker = OpenAttack.attackers.DeepWordBugAttacker()

        elif attack_name == 'fd':
            attacker = OpenAttack.attackers.FDAttacker()

        elif attack_name == 'gan':
            attacker = OpenAttack.attackers.GANAttacker()

        elif attack_name == 'genetic':
            attacker = OpenAttack.attackers.GeneticAttacker()

        elif attack_name == 'hotflip':
            attacker = OpenAttack.attackers.HotFlipAttacker()

        elif attack_name == 'pso':
            attacker = OpenAttack.attackers.PSOAttacker()

        elif attack_name == 'pwws':
            attacker = OpenAttack.attackers.PWWSAttacker()

        elif attack_name == 'textbugger':
            attacker = OpenAttack.attackers.TextBuggerAttacker()

        elif attack_name == 'textfooler':
            attacker = OpenAttack.attackers.TextFoolerAttacker()

        elif attack_name == 'uat':
            attacker = OpenAttack.attackers.UATAttacker()

        elif attack_name == 'viper':
            attacker = OpenAttack.attackers.VIPERAttacker()

        else:
            raise ValueError('unknown attack {}'.format(attack_name))

        if skip_words is not None and 'skip_words' in attacker.config:
            attacker.config['skip_words'].update(set(skip_words))

    else:
        raise ValueError('Unknown attacker {}!'.format(attack_name))

    return attacker


def batch_encode(tokenizer, text_list):
    """
    Encode list of text into lists of tokens.
    """
    if hasattr(tokenizer, "batch_encode"):
        result = tokenizer.batch_encode(text_list)
    else:
        result = [tokenizer.encode(text_input) for text_input in text_list]
    return result


def generate_multiclass_roc_curves(y_true, y_score, class_names=None):
    """
    Returns a dictionary of One vs. Rest ROC curves. Also includes
    a macro ROC curve.

    Input
    y_true: 1d arry of class label integers
    y_score: 2d array of shape=(no. samples, no. classes)
    label_map: 1d list of class names.
    """

    # binarize the output
    n_classes = y_score.shape[1]
    y_true = label_binarize(y_true, classes=list(range(n_classes)))

    # create class names if None
    if class_names is None:
        class_names = ['class_{}'.format(i) for i in range(n_classes)]

    # compute ROC curve and ROC area for each class
    roc_curves = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_curves[class_names[i]] = (fpr, tpr, None)

    # first aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr for k, (fpr, tpr, _) in roc_curves.items()]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for k, (fpr, tpr, _) in roc_curves.items():
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    # finally average it
    mean_tpr /= n_classes
    roc_curves['Macro Average'] = (all_fpr, mean_tpr, None)

    return roc_curves


def plot_roc_curves(curves, ax=None, zoom=False, width=18, legend_fontsize=7.5):
    """
    Plot ROC curve.
    """
    golden_ratio = 1.61803

    if ax is None:
        fig, ax = plt.figure(figsize=(width, width / golden_ratio))

    ax.set_title('ROC curves')

    ax.set_ylabel("True Positive Rate")
    ax.set_ylim([-0.05, 1.05])
    ax.set_yticks(np.arange(0, 1, 0.1), minor=True)

    ax.set_xlabel("False Positive Rate")
    ax.set_xticks(np.arange(0, 1, 0.1), minor=True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    if zoom:
        ax.set_xlim([0.0, 0.01])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.001))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.0001))

    ax.plot([0, 1], [0, 1], "k:", label="Random")
    for name, (fpr, tpr, thresholds) in curves.items():
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label='{}: {:.3f}'.format(name, auc_score))

    ax.legend(loc="lower right", fontsize=legend_fontsize)
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor', linewidth=0.1)

    return ax


def plot_confusion_matrix(cm,
                          target_names,
                          cmap=None,
                          normalize=True,
                          ax=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if ax is None:
        fig, ax = plt.figure(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names, rotation=45, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, '{:.1f}'.format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            ax.text(j, i, '{:.1f}'.format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
