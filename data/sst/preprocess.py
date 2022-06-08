import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main(random_state=42):

    # read in data
    print('reading data...')
    df = pd.read_csv('SST-2/train.tsv', sep='\t')
    val_df = pd.read_csv('SST-2/dev.tsv', sep='\t')

    # rename columns
    print('renaming columns...')
    df = df.rename(columns={'sentence': 'text'})
    val_df = val_df.rename(columns={'sentence': 'text'})

    print('split data...')
    train_df, test_df = train_test_split(df, test_size=0.5, stratify=df['label'],
                                                     random_state=random_state)

    # display datasets
    print('\nTrain:\n{}'.format(train_df))
    print('Percent positive in training set: {:.2f}%'.format(len(train_df[train_df['label'] == 1]) / len(train_df) *
                                                            100))

    print('\nValidation:\n{}'.format(val_df))
    print('Percent positive in validation set: {:.2f}%'.format(len(val_df[val_df['label'] == 1]) / len(val_df) * 100))

    print('\nTest:\n{}'.format(test_df))
    print('Percent positive in testing set: {:.2f}%'.format(len(test_df[test_df['label'] == 1]) / len(test_df) * 100))

    exit()
    print('saving datasets...')
    columns = ['text', 'label']

    out_dir = 'sst-2/'
    path = os.path.join(out_dir)
    os.mkdir(path)

    train_df.to_csv(os.path.join(path, 'train.csv'), index=None, columns=columns)
    val_df.to_csv(os.path.join(path, 'val.csv'), index=None, columns=columns)
    test_df.to_csv(os.path.join(path, 'test.csv'), index=None, columns=columns)






if __name__ == '__main__':
    main()
