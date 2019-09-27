import numpy as np
import os
import pandas as pd
import hashlib
import mrecsys
import time
from mrecsys.utils.indexer import Indexer
from spotlight.interactions import Interactions


def hash_time(_time):
    return hashlib.sha1(str(_time).encode('utf-8')).hexdigest()[:10]


def reload_data(user_col='user_id',
                item_col='item_id',
                time_col='timestamp',
                weight_col='weight'):
    """
    some code that crawl interactions data from database
    saving and indexing them in dataset directory:
        - dicts: id and indexed id of users and items
        - interactions.csv: interactions dataframe of indexed users and items
    """
    # dummy case
    df = pd.read_csv('https://storage.cloud.google.com/dimo_project/interactions.csv')
    df = df[[time_col, user_col, item_col, weight_col]]

    user_dict = Indexer(df[user_col].unique())
    item_dict = Indexer(df[item_col].unique())

    df[user_col] = df[user_col].apply(lambda x: user_dict.index(x))
    df[item_col] = df[item_col].apply(lambda x: item_dict.index(x))

    current_time = int(time.time())
    user_dict.dumps(os.path.join(mrecsys.__dataset_path__, 'dicts/user_to_index/{}.json'.format(current_time)))
    item_dict.dumps(os.path.join(mrecsys.__dataset_path__, 'dicts/item_to_index/{}.json'.format(current_time)))

    df = df.rename(
        columns={
            user_col: 'user_id',
            item_col: 'item_id',
            time_col: 'timestamp',
            weight_col: 'weight'
        })
    df['user_id'] = df['user_id'].astype(np.int32)
    df['item_id'] = df['item_id'].astype(np.int32)
    df['timestamp'] = df['timestamp'].astype(np.int32)
    df['weight'] = df['weight'].astype(np.float32)

    df.to_csv(os.path.join(mrecsys.__dataset_path__,
                           'interactions/interactions_{}.csv'.format(current_time)),
              index=False)


def load_interactions(path):
    try:
        df = pd.read_csv(path)
        interactions = Interactions(user_ids=df['user_id'].astype(np.int32).values,
                                    item_ids=df['item_id'].astype(np.int32).values,
                                    timestamps=df['timestamp'].astype(np.int32).values,
                                    weights=df['weight'].astype(np.int32).values)
        reloaded_time = int(os.path.basename(path).split('.')[0])

        user_indexer = Indexer(dumped_filepath=os.path.join(mrecsys.__dataset_path__,
                                                            'dicts/user_to_index/{}.json'.format(reloaded_time)))
        item_indexer = Indexer(dumped_filepath=os.path.join(mrecsys.__dataset_path__,
                                                            'dicts/item_to_index/{}.json'.format(reloaded_time)))
        return interactions, reloaded_time, user_indexer, item_indexer
    except:
        return None


def load_interactions_by_time(time_code):
    try:
        path = os.path.join(mrecsys.__dataset_path__, 'interactions/{}.csv'.format(time_code))
        interactions, time_code,  user_indexer, item_indexer = load_interactions(path)
        return interactions, user_indexer, item_indexer
    except:
        return None


def load_latest_interactions():
    from os import listdir
    from os.path import isfile, join

    dirpath = os.path.join(mrecsys.__dataset_path__, 'interactions')
    files = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    if len(files) > 0:
        path = os.path.join(dirpath, files[-1])
        print(path)
        return load_interactions(path)
    return None


if __name__ == '__main__':
    reload_data(item_col='service_id')
