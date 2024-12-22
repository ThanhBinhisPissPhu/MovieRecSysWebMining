import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_data_ml100k(random_state = 1, test_size = 0.1):
    columns_name=['user_id','item_id','rating','timestamp']
    df = pd.read_csv("data/ml-100k/u.data",sep="\t",names=columns_name)

    train, test = train_test_split(df.values, test_size=test_size, random_state = random_state)
    train = pd.DataFrame(train, columns = df.columns)
    test = pd.DataFrame(test, columns = df.columns)
    le_user = preprocessing.LabelEncoder()
    le_item = preprocessing.LabelEncoder()
    train['user_id_idx'] = le_user.fit_transform(train['user_id'].values)
    train['item_id_idx'] = le_item.fit_transform(train['item_id'].values)

    train_user_ids = train['user_id'].unique()
    train_item_ids = train['item_id'].unique()

    test = test[(test['user_id'].isin(train_user_ids)) & (test['item_id'].isin(train_item_ids))]

    test['user_id_idx'] = le_user.transform(test['user_id'].values)
    test['item_id_idx'] = le_item.transform(test['item_id'].values)

    return train, test

import numpy as np

def load_data_ml100k_cs(N = 200, K = 20):
    columns_name=['user_id','item_id','rating','timestamp']
    df = pd.read_csv("data/ml-100k/u.data",sep="\t",names=columns_name)

    user2count = df.groupby(['item_id']).size().reset_index(name='count').sort_values(by='count')
    item_ids = list(user2count['item_id'])
    counts = np.array(user2count['count'])

    item_ids, counts = np.asarray(item_ids), np.asarray(counts)
    hot_item_ids = item_ids[counts > N]
    cold_item_ids = item_ids[np.logical_and(counts <= N, counts >= 3 * K)]
    item_group = df.groupby('item_id')
    
    train_base = pd.DataFrame()
    for item_id in hot_item_ids:
        df_hot = item_group.get_group(item_id).sort_values(by='timestamp')
        train_base = pd.concat([train_base, df_hot], ignore_index=True)

    train_warm_a, train_warm_b, train_warm_c, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for item_id in cold_item_ids:
        df_cold = item_group.get_group(item_id).sort_values(by='timestamp')
        train_warm_a = pd.concat([train_warm_a, df_cold[:K]], ignore_index=True)
        train_warm_b = pd.concat([train_warm_b, df_cold[K:2*K]], ignore_index=True)
        train_warm_c = pd.concat([train_warm_c, df_cold[2*K:3*K]], ignore_index=True)
        test = pd.concat([test, df_cold[3*K:]], ignore_index=True)

    train_base = pd.DataFrame(train_base, columns = df.columns)
    train_warm_a = pd.DataFrame(train_warm_a, columns = df.columns)
    train_warm_b = pd.DataFrame(train_warm_b, columns = df.columns)
    train_warm_c = pd.DataFrame(train_warm_c, columns = df.columns)
    test = pd.DataFrame(test, columns = df.columns)

    le_user = preprocessing.LabelEncoder()
    le_item = preprocessing.LabelEncoder()

    le_user.fit(np.unique(all_user_ids))
    le_item.fit(np.unique(all_item_ids))

    # Transform the user_id and item_id columns for each dataset
    train_base['user_id_idx'] = le_user.transform(train_base['user_id'].values)
    train_base['item_id_idx'] = le_item.transform(train_base['item_id'].values)
    train_warm_a['user_id_idx'] = le_user.transform(train_warm_a['user_id'].values)
    train_warm_a['item_id_idx'] = le_item.transform(train_warm_a['item_id'].values)
    train_warm_b['user_id_idx'] = le_user.transform(train_warm_b['user_id'].values)
    train_warm_b['item_id_idx'] = le_item.transform(train_warm_b['item_id'].values)
    train_warm_c['user_id_idx'] = le_user.transform(train_warm_c['user_id'].values)
    train_warm_c['item_id_idx'] = le_item.transform(train_warm_c['item_id'].values)


    # train_base['user_id_idx'] = le_user.fit_transform(train_base['user_id'].values)
    # train_base['item_id_idx'] = le_item.fit_transform(train_base['item_id'].values)
    # train_warm_a['user_id_idx'] = le_user.fit_transform(train_warm_a['user_id'].values)
    # train_warm_a['item_id_idx'] = le_item.fit_transform(train_warm_a['item_id'].values)
    # train_warm_b['user_id_idx'] = le_user.fit_transform(train_warm_b['user_id'].values)
    # train_warm_b['item_id_idx'] = le_item.fit_transform(train_warm_b['item_id'].values)
    # train_warm_c['user_id_idx'] = le_user.fit_transform(train_warm_c['user_id'].values)
    # train_warm_c['item_id_idx'] = le_item.fit_transform(train_warm_c['item_id'].values)

    all_user_ids = np.concatenate([
        train_base['user_id_idx'].values,
        train_warm_a['user_id_idx'].values,
        train_warm_b['user_id_idx'].values,
        train_warm_c['user_id_idx'].values
    ])
    train_user_ids = np.unique(all_user_ids)

    # Collect all unique item IDs
    all_item_ids = np.concatenate([
        train_base['item_id_idx'].values,
        train_warm_a['item_id_idx'].values,
        train_warm_b['item_id_idx'].values,
        train_warm_c['item_id_idx'].values
    ])
    train_item_ids = np.unique(all_item_ids)

    # Optionally, convert to lists if needed
    train_user_ids = train_user_ids.tolist()
    train_item_ids = train_item_ids.tolist()

    # train_user_ids = train_base['user_id'].unique()
    # train_item_ids = train_base['item_id'].unique()

    test = test[(test['user_id'].isin(train_user_ids)) & (test['item_id'].isin(train_item_ids))]

    test['user_id_idx'] = le_user.transform(test['user_id'].values)
    test['item_id_idx'] = le_item.transform(test['item_id'].values)

    return train_base, train_warm_a, train_warm_b, train_warm_c, test


if __name__ == "__main__":
    train, test = load_data_ml100k_cs(random_state=1)
    for df in [train,test]:
        print("{} size: {}".format(df, len(df)))