import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Helper function to calculate MSE and MAE
def calculate_metrics(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    return mse, mae

# Data loading and preparation
def load_and_process_data():
    # Load the Ratings data
    data = pd.read_csv('ml-100k/u.data', sep="\t", header=None)
    data.columns = ['user id', 'movie id', 'rating', 'timestamp']
    
    # Load the User data
    users = pd.read_csv('ml-100k/u.user', sep="|", encoding='latin-1', header=None)
    users.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']
    
    # Load Movie data
    items = pd.read_csv('ml-100k/u.item', sep="|", encoding='latin-1', header=None)
    items.columns = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 
                     'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                     'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Merge datasets
    dataset = data.merge(users, on='user id', how='left').merge(items, on='movie id', how='left')

    # Encode gender as binary
    dataset['gender'] = (dataset['gender'] == 'M').astype(int)  

    # Encode occupation as integers
    label_encoder = LabelEncoder()
    dataset['occupation'] = label_encoder.fit_transform(dataset['occupation'])  

    # Convert age into intervals (bins) and encode as integers
    bins = [0, 18, 25, 35, 45, 50, 60, 100]
    labels = [0, 1, 2, 3, 4, 5, 6]
    dataset['age'] = pd.cut(dataset['age'], bins=bins, labels=labels).astype(int)

    # Drop irrelevant columns
    dataset.drop(['zip code', 'movie title', 'release date', 'video release date', 'IMDb URL'], axis=1, inplace=True)

    # Split into train, validation, and test sets
    train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Separate features (X) and target (y)
    X_train, y_train = train_data.drop(columns=['rating','user id','movie id','timestamp']), train_data['rating']
    X_valid, y_valid = valid_data.drop(columns=['rating','user id','movie id','timestamp']), valid_data['rating']
    X_test, y_test = test_data.drop(columns=['rating','user id','movie id','timestamp']), test_data['rating']
    

    return X_train, y_train, X_valid, y_valid, X_test, y_test
class MovieLensDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return {col: torch.tensor(self.X.iloc[idx][col]) for col in self.X.columns}, torch.tensor(self.y.iloc[idx])
# DeepFM Model
class DeepFM(nn.Module):
    def __init__(self, num_occupations, num_age_bins, num_genres, embed_size):
        super().__init__()

        # Embedding layers for individual features
        self.occupation_embedding = nn.Embedding(num_occupations, embed_size, device=device)
        self.age_embedding = nn.Embedding(num_age_bins, embed_size, device=device)
        self.genre_embedding = nn.Embedding(num_genres, embed_size, device=device)

        # Linear layer for wide features
        self.linear_layer = nn.Linear(embed_size * 2 + num_genres, 1, device=device)

        # Deep component
        self.deep_stack = nn.Sequential(
            nn.Linear(embed_size * 2 + num_genres, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 64, device=device),
            nn.ReLU(),
            nn.Linear(64, 32, device=device),
            nn.ReLU()
        )
        self.output = nn.Linear(32 + 1, 1, device=device)  # Combine FM and Deep parts

    def forward(self, x_train):

        # Embedding lookup for each feature
        joint = []
        age_emb = self.age_embedding(x_train['age']) #batch_size , emb_size
        occ_emb = self.occupation_embedding(x_train['occupation'])
        for key in x_train.keys():
            if key not in ['age', 'occupation']:
                joint.append(x_train[key])
        joint = torch.stack(joint, dim=1) # batch_size, nb_attr

        cat_joint = torch.cat([age_emb, occ_emb, joint], dim=1) # batch_size, nb_attr + 2 * emb_size


        # Linear and FM part
        fm_out = self.linear_layer(cat_joint)

        # Deep part
        deep_out = self.deep_stack(cat_joint)

        # Final output
        out = self.output(torch.cat((deep_out, fm_out), dim=1))
        return out.squeeze(1)

# Training and evaluation loop
def train_deepfm(model:DeepFM, train_loader, val_loader, test_loader, learning_rate=1e-3, weight_decay=0.01, num_epochs=10):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, valid_losses = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()

        for batch in  train_loader :
            x_train ,y_train = batch[0], batch[1]
            y_pred_train = model(x_train)
            loss_train = loss_fn(y_pred_train, y_train.float())
            print(f"loss : {loss_train.item()}")
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        # # Validation
        # model.eval()
        # with torch.no_grad():
        #     y_pred_valid = model(**X_valid)
        #     loss_valid = loss_fn(y_pred_valid, y_valid.float())

        # train_losses.append(loss_train.item())
        # valid_losses.append(loss_valid.item())

        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Valid Loss: {loss_valid.item():.4f}")

    # Testing
    # model.eval()
    # with torch.no_grad():
    #     y_pred_test = model(**X_test)
    #     test_mse, test_mae = calculate_metrics(y_test.float(), y_pred_test)
    #     print(f"Test Results - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
    return model

# Main execution
X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_process_data()


# Create DataLoader for train, validation, and test sets
train_dataset = MovieLensDataset(X_train, y_train)
valid_dataset = MovieLensDataset(X_valid, y_valid)
test_dataset = MovieLensDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


deepfm_model = DeepFM(
    num_occupations=X_train['occupation'].nunique(),
    num_age_bins=7,  # Based on age intervals
    num_genres=20,  # Number of genres in MovieLens dataset
    embed_size=16
)

trained_model = train_deepfm(deepfm_model, train_loader, valid_loader, test_loader, num_epochs=20)