import os
import re
import string
import time

# for word of bags approach -- remove?
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def stats(df):
    print('\nTotal no. samples: {:,}'.format(len(df)))
    print('Percent news: {:.2f}%'.format(len(df[df['label'] == 3]) / len(df) * 100))
    print('Percent existence: {:.2f}%'.format(len(df[df['label'] == 2]) / len(df) * 100))
    print('Percent neutral: {:.2f}%'.format(len(df[df['label'] == 1]) / len(df) * 100))
    print('Percent non-existence: {:.2f}%'.format(len(df[df['label'] == 0]) / len(df) * 100))

def clean(df):
    # lowercase all letters
    df['message'] = df['message'].apply(lambda x: x.lower())

    # remove urls
    df['message'] = df['message'].apply(lambda x: re.sub(r'http\S+', '', x))

    # remove retweet token
    df['message'] = df['message'].apply(lambda x: re.sub(r'^rt', '', x))

    # remove non-ascii tokens
    df['message'] = df['message'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
    
    # anonymize twitter usernames
    df['message'] = df['message'].apply(lambda x: re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '@user', x))

    # remove hashtags
    df['message'] = df['message'].apply(lambda x: re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)', '', x))
    
    # remove punctuation
    #df['message'] = df['message'].apply(lambda x: re.sub(r'[{}]'.format(string.punctuation), '', x))
    
    return df

def main(random_state=1):
    start = time.time()
    # read in data
    print('reading data...')
    in_dir = ''
    df = pd.read_csv(os.path.join(in_dir, 'twitter_sentiment_data.csv'), engine='python')
    df['sentiment'] = df['sentiment'].apply(lambda x: x + 1)

    df = df.loc[df.sentiment != 3]    

    print('cleaning data...')
    df = clean(df)
    # remove only whitespace rows
    df = df.query('not message.str.isspace()')    

    # rename columns
    df = df.rename(columns={'sentiment': 'label', 'message': 'text'})

    # dataset statistics
    stats(df)

    # compute average number of words/chars per sample
    df['word_len'] = df['text'].apply(lambda x: len(x.split()))
    df['char_len'] = df['text'].apply(lambda x: len(x))
    print('\n{}'.format(df['word_len'].describe()))
    print('\n{}'.format(df['char_len'].describe()))

    # split data into train, val, and test
    print('splitting dataset into train, val, and test...')
    train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df['label'])
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=random_state, stratify=val_test_df['label'])

    # display datasets
    print('\nTrain:\n{}'.format(train_df))
    stats(train_df)

    print('\nValidation:\n{}'.format(val_df))
    stats(val_df)

    print('\nTest:\n{}'.format(test_df))
    stats(test_df)

    # save datasets
    columns = ['text', 'label']
    print('\nsaving datasets...')
    train_df.to_csv('train.csv', index=None, columns=columns)
    val_df.to_csv('val.csv', index=None, columns=columns)
    test_df.to_csv('test.csv', index=None, columns=columns)

    end = time.time()
    print(f'Elapsed time: {end - start:.2f}s')


if __name__ == '__main__':
    main()

