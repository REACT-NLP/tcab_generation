from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def main(random_state=42):

    # load dataset from the HuggingFace Hub
    dataset = load_dataset("imdb")


    # convert to dataframes, and prepare the data for training
    tv_df = pd.DataFrame.from_dict(dataset['train'])
    test_df = pd.DataFrame.from_dict(dataset['test'])
    train_df, val_df = train_test_split(tv_df, test_size=0.2, stratify=tv_df['label'],
                                        random_state=random_state)


    # train and validation sets statistics
    tv_df['word_len'] = tv_df['text'].apply(lambda x: len(x.split()))
    print('\nTotal no. samples: {:,}'.format(len(tv_df)))
    print('Percent negative: {:.2f}%'.format(len(tv_df[tv_df['label'] == 0]) / len(tv_df) * 100))
    print('Percent positive: {:.2f}%'.format(len(tv_df[tv_df['label'] == 1]) / len(tv_df) * 100))
    print('\n{}'.format(tv_df['word_len'].describe()))


    # test set statistics
    test_df['word_len'] = test_df['text'].apply(lambda x: len(x.split()))
    print('\nTotal no. samples: {:,}'.format(len(test_df)))
    print('Percent negative: {:.2f}%'.format(len(test_df[test_df['label'] == 0]) / len(test_df) * 100))
    print('Percent positive: {:.2f}%'.format(len(test_df[test_df['label'] == 1]) / len(test_df) * 100))
    print('\n{}'.format(test_df['word_len'].describe()))


    # display datasets
    print('\nTrain:\n{}'.format(train_df))
    print('Percent toxic: {:.2f}%'.format(len(train_df[train_df['label'] == 1]) / len(train_df) * 100))

    print('\nValidation:\n{}'.format(val_df))
    print('Percent toxic: {:.2f}%'.format(len(val_df[val_df['label'] == 1]) / len(val_df) * 100))

    print('\nTest:\n{}'.format(test_df))
    print('Percent toxic: {:.2f}%'.format(len(test_df[test_df['label'] == 1]) / len(test_df) * 100))

    print('saving datasets...')
    columns = ['text', 'label']
    # out_dir = '../data/imdb/'
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    train_df.to_csv(('train.csv'), index=None, columns=columns)
    val_df.to_csv(('val.csv'), index=None, columns=columns)
    test_df.to_csv(('test.csv'), index=None, columns=columns)

if __name__ == '__main__':
    main()
