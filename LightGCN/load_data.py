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