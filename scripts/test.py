import os
import gc
gc.collect()

# set environments
os.environ['TA_CACHE_DIR'] = '.cache'  # TextAttack cache
os.environ['TRANSFORMERS_CACHE'] = '.cache'  # transformers cache

import torch
torch.cuda.empty_cache()
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from scipy.special import softmax

import utility

def test(model, test_dataloader, y_test):
    """
    Evaluate performance of the model on a
    held-out test set.
    """
    # result container
    result = {}

    # activate evaluation mode
    model.eval()

    # generate predictions on the test set
    all_preds = []
    for text_list, labels in test_dataloader:
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
        result['recall'] = recall_score(y_test, y_pred)
    else:
        result['auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
        result['ap'] = -1
        result['f1'] = f1_score(y_test, y_pred, average='macro')
        result['recall'] = recall_score(y_test, y_pred, average='macro')

    # save predictions
    result['pred'] = y_pred
    result['proba'] = y_proba

    return result
 
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data_path = os.path.join('../', args.test_dataset, 'data.csv')
    print(test_data_path)
    test_df = pd.read_csv(test_data_path)

    text = np.array(test_df['text'].values)
    label = np.array(test_df['label'].values)
    test_data = list(zip(text, label))

    model = utility.get_model(model_name=args.target_model)
    target_model_path = os.path.join('target_models', args.target_model_dataset, args.target_model, 'weights.pt')
    print(target_model_path)
    model.load_state_dict(torch.load(target_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=32)
    y_test = label
    res = test(model, test_dataloader, y_test)
    print(res)

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model', type=str, default='bert', help='target model name, must be the same as dir in dropbox')
    parser.add_argument('--target_model_dataset', type=str, default='hatebase', help='target model dataset name, must be the same as dir in dropbox')
    parser.add_argument('--test_dataset', type=str, default='wikipedia_aggression', help='test dataset name, must be the same as dir in dropbox')
    args = parser.parse_args()
    main(args)

