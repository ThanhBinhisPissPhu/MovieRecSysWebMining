import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import random
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from model import LightGCN
from utils import *

if __name__ == "__main__":
    columns_name=['user_id','item_id','rating','timestamp']
    df = pd.read_csv("data/ml-100k/u.data",sep="\t",names=columns_name)

    train, test = train_test_split(df.values, test_size=0.1, random_state = 1)
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
    
    n_users = train['user_id_idx'].nunique()
    n_items = train['item_id_idx'].nunique()

    latent_dim = 64
    n_layers = 3  

    lightGCN = LightGCN(train, n_users, n_items, n_layers, latent_dim)

    optimizer = torch.optim.Adam(lightGCN.parameters(), lr = 0.005)

    EPOCHS = 30
    BATCH_SIZE = 1024 
    DECAY = 0.0001
    K = 10
    loss_list_epoch = []
    MF_loss_list_epoch = []
    reg_loss_list_epoch = []

    recall_list = []
    precision_list = []
    ndcg_list = []
    map_list = []

    train_time_list = []
    eval_time_list = [] 

    for epoch in tqdm(range(EPOCHS)):
        n_batch = int(len(train)/BATCH_SIZE)
    
        final_loss_list = []
        MF_loss_list = []
        reg_loss_list = []
    
        best_ndcg = -1
    
        train_start_time = time.time()
        lightGCN.train()
        for batch_idx in range(n_batch):

            optimizer.zero_grad()

            users, pos_items, neg_items = data_loader(train, BATCH_SIZE, n_users, n_items)

            users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = lightGCN.forward(users, pos_items, neg_items)

            mf_loss, reg_loss = bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)
            reg_loss = DECAY * reg_loss
            final_loss = mf_loss + reg_loss

            final_loss.backward()
            optimizer.step()

            final_loss_list.append(final_loss.item())
            MF_loss_list.append(mf_loss.item())
            reg_loss_list.append(reg_loss.item())


        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        lightGCN.eval()
        with torch.no_grad():
        
            final_user_Embed, final_item_Embed, initial_user_Embed,initial_item_Embed = lightGCN.propagate_through_layers()
            test_topK_recall,  test_topK_precision, test_topK_ndcg, test_topK_map  = get_metrics(final_user_Embed, final_item_Embed, n_users, n_items, train, test, K)


        if test_topK_ndcg > best_ndcg:
            best_ndcg = test_topK_ndcg
        
            torch.save(final_user_Embed, 'checkpoint/GCN/final_user_Embed.pt')
            torch.save(final_item_Embed, 'checkpoint/GCN/final_item_Embed.pt')
            torch.save(initial_user_Embed, 'checkpoint/GCN/initial_user_Embed.pt')
            torch.save(initial_item_Embed, 'checkpoint/GCN/initial_item_Embed.pt')
        

        eval_time = time.time() - train_end_time

        loss_list_epoch.append(round(np.mean(final_loss_list),4))
        MF_loss_list_epoch.append(round(np.mean(MF_loss_list),4))
        reg_loss_list_epoch.append(round(np.mean(reg_loss_list),4))

        recall_list.append(round(test_topK_recall,4))
        precision_list.append(round(test_topK_precision,4))
        ndcg_list.append(round(test_topK_ndcg,4))
        map_list.append(round(test_topK_map,4))

        train_time_list.append(train_time)
        eval_time_list.append(eval_time)