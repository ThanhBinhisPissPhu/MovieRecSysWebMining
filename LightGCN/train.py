import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from model import LightGCN
from load_data import load_data_ml100k
from utils import *
import os

if __name__ == "__main__":
    train, test = load_data_ml100k(random_state = 1, test_size = 0.15)
    
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

    directory = "checkpoint/GCN"
    os.makedirs(directory, exist_ok=True)

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
        
            torch.save(final_user_Embed, f'{directory}/final_user_Embed.pt')
            torch.save(final_item_Embed, f'{directory}/final_item_Embed.pt')
            torch.save(initial_user_Embed, f'{directory}/initial_user_Embed.pt')
            torch.save(initial_item_Embed, f'{directory}/initial_item_Embed.pt')
        

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


    epoch_list = [(i+1) for i in range(EPOCHS)]
    plt.plot(epoch_list, recall_list, label='Recall')
    plt.plot(epoch_list, precision_list, label='Precision')
    plt.plot(epoch_list, ndcg_list, label='NDCG')
    plt.plot(epoch_list, map_list, label='MAP')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    save_path = os.path.join(directory, "metrics_plot.png")
    plt.savefig(save_path)

    print("Last Epoch's Test Data Recall -> ", recall_list[-1])
    print("Last Epoch's Test Data Precision -> ", precision_list[-1])
    print("Last Epoch's Test Data NDCG -> ", ndcg_list[-1])
    print("Last Epoch's Test Data MAP -> ", map_list[-1])

    print("Last Epoch's Train Data Loss -> ", loss_list_epoch[-1])