"""
Preprocess dataset, adapted from:
https://github.com/conversationai/unintended-ml-bias-analysis/blob/master/unintended_ml_bias/Prep_Wikipedia_Data.ipynb

Predefined train/val/test split.

Simple Model:
https://www.kaggle.com/thousandvoices/simple-lstm
Batch Size: 128
Epochs: 10
Max. Sequence Length: 250
Learning Rate: 0.00005
"""
import os

import pandas as pd


def main(random_state=1):

    # read in data
    print('reading data...')
    in_dir = ''
    toxicity_annotated_comments = pd.read_csv(os.path.join(in_dir, 'toxicity_annotated_comments.tsv'), sep='\t')
    toxicity_annotations = pd.read_csv(os.path.join(in_dir, 'toxicity_annotations.tsv'), sep='\t')

    # attach labels
    annotations_gped = toxicity_annotations.groupby('rev_id', as_index=False).agg({'toxicity': 'mean'})
    df = pd.merge(annotations_gped, toxicity_annotated_comments, on='rev_id')

    # remove newline and tab tokens
    df['comment'] = df['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    df['comment'] = df['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

    # binarize labels
    print('binarizing labels...')
    df['label'] = df['toxicity'].apply(lambda x: 1 if x > 0.5 else 0)

    # rename columns
    print('renaming columns...')
    df = df.rename(columns={'comment': 'text'})

    # dataset statistics
    print('\nTotal no. samples: {:,}'.format(len(df)))
    print('Percent toxic: {:.2f}%'.format(len(df[df['label'] == 1]) / len(df) * 100))

    # compute average number of words/chars per sample
    df['word_len'] = df['text'].apply(lambda x: len(x.split()))
    df['char_len'] = df['text'].apply(lambda x: len(x))
    print('\n{}'.format(df['word_len'].describe()))
    print('\n{}'.format(df['char_len'].describe()))

    # split data into train, val, and test
    print('splitting dataset into train, val, and test...')
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'dev']
    test_df = df[df['split'] == 'test']

    # display datasets
    print('\nTrain:\n{}'.format(train_df))
    print('Percent toxic: {:.2f}%'.format(len(train_df[train_df['label'] == 1]) / len(train_df) * 100))

    print('\nValidation:\n{}'.format(val_df))
    print('Percent toxic: {:.2f}%'.format(len(val_df[val_df['label'] == 1]) / len(val_df) * 100))

    print('\nTest:\n{}'.format(test_df))
    print('Percent toxic: {:.2f}%'.format(len(test_df[test_df['label'] == 1]) / len(test_df) * 100))

    # save datasets
    columns = ['text', 'label']
    print('\nsaving datasets...')
    train_df.to_csv('train.csv', index=None, columns=columns)
    val_df.to_csv('val.csv', index=None, columns=columns)
    test_df.to_csv('test.csv', index=None, columns=columns)


if __name__ == '__main__':
    main()
