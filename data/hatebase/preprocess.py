"""
Preprocess dataset.

No previous preprocessing exists.

Dataset is split into an 80/10/10 train/val/test split
using stratified sampling.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main(test_frac=0.2, random_state=1):

    # read in data
    print('reading data...')
    df = pd.read_csv(os.path.join('hate-speech-and-offensive-language-master',
                                  'data', 'labeled_data.csv'))

    # rename columns
    df = df.rename(columns={'tweet': 'text'})

    # combine hate speech and offensive messages
    df['label'] = df['class'].apply(lambda x: 1 if x <= 1 else 0)

    # all sample statistics
    df['word_len'] = df['text'].apply(lambda x: len(x.split()))
    print('\nTotal no. samples: {:,}'.format(len(df)))
    print('Percent toxic: {:.2f}%'.format(len(df[df['label'] == 1]) / len(df) * 100))
    print('\n{}'.format(df['word_len'].describe()))

    # creating train and test sets
    print('creating train, val, and test splits...')
    train_df, val_test_df = train_test_split(df, test_size=0.2, stratify=df['label'],
                                             random_state=random_state)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, stratify=val_test_df['label'],
                                       random_state=random_state)

    # display datasets
    print('\nTrain:\n{}'.format(train_df))
    print('Percent toxic: {:.2f}%'.format(len(train_df[train_df['label'] == 1]) / len(train_df) * 100))

    print('\nValidation:\n{}'.format(val_df))
    print('Percent toxic: {:.2f}%'.format(len(val_df[val_df['label'] == 1]) / len(val_df) * 100))

    print('\nTest:\n{}'.format(test_df))
    print('Percent toxic: {:.2f}%'.format(len(test_df[test_df['label'] == 1]) / len(test_df) * 100))

    # save datasets
    print('saving datasets...')
    columns = ['text', 'label']
    train_df.to_csv('train.csv', index=None, columns=columns)
    val_df.to_csv('val.csv', index=None, columns=columns)
    test_df.to_csv('test.csv', index=None, columns=columns)


if __name__ == '__main__':
    main()
