from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def main(random_state=42):

    # load dataset from the HuggingFace Hub
    dataset = load_dataset('sst', 'default')


    # convert to dataframes, and prepare the data for training
    train_df = pd.DataFrame.from_dict(dataset['train'])
    test_df = pd.DataFrame.from_dict(dataset['test'])
    val_df = pd.DataFrame.from_dict(dataset['validation'])

    # binarize labels
    print('binarizing labels...')
    train_df['label'] = train_df['label'].apply(lambda x: 1 if x > 0.5 else 0)
    val_df['label'] = val_df['label'].apply(lambda x: 1 if x > 0.5 else 0)
    test_df['label'] = test_df['label'].apply(lambda x: 1 if x > 0.5 else 0)

    # rename columns
    print('renaming columns...')
    train_df = train_df.rename(columns={'sentence': 'text'})
    val_df = val_df.rename(columns={'sentence': 'text'})
    test_df = test_df.rename(columns={'sentence': 'text'})


    # display datasets
    print('\nTrain:\n{}'.format(train_df))
    print('Percent positive in training set: {:.2f}%'.format(len(train_df[train_df['label'] == 1]) / len(train_df) *
                                                            100))

    print('\nValidation:\n{}'.format(val_df))
    print('Percent positive in validation set: {:.2f}%'.format(len(val_df[val_df['label'] == 1]) / len(val_df) * 100))

    print('\nTest:\n{}'.format(test_df))
    print('Percent positive in testing set: {:.2f}%'.format(len(test_df[test_df['label'] == 1]) / len(test_df) * 100))

    print('saving datasets...')
    columns = ['text', 'label']

    train_df.to_csv(('train.csv'), index=None, columns=columns)
    val_df.to_csv(('val.csv'), index=None, columns=columns)
    test_df.to_csv(('test.csv'), index=None, columns=columns)

if __name__ == '__main__':
    main()
