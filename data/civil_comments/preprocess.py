"""
Preprocess dataset.

80/10/10 split using the training data only:
https://arxiv.org/pdf/2004.14088.pdf

Simple Model:
https://www.kaggle.com/thousandvoices/simple-lstm
Batch Size: 256 or 512
Epochs: 10
Max. Sequence Length: 225
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main(random_state=1):

    # read in data
    print('reading data...')
    in_dir = 'jigsaw-unintended-bias-in-toxicity-classification'
    df = pd.read_csv(os.path.join(in_dir, 'train.csv'))

    # binarize labels
    print('binarizing labels...')
    df['label'] = df['target'].apply(lambda x: 1 if x >= 0.5 else 0)

    # rename columns
    print('renaming columns...')
    df = df.rename(columns={'comment_text': 'text'})

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
    train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df['label'])
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=random_state,
                                       stratify=val_test_df['label'])

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
