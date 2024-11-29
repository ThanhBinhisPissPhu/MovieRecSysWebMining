import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    X_train, y_train = train_data.drop(columns=['rating']), train_data['rating']
    X_valid, y_valid = valid_data.drop(columns=['rating']), valid_data['rating']
    X_test, y_test = test_data.drop(columns=['rating']), test_data['rating']

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# DeepFM Model
class DeepFM(nn.Module):
    def __init__(self, num_users, num_movies, num_genders, num_occupations, num_age_bins, num_genres, embed_size):
        super().__init__()

        # Embedding layers for individual features
        self.user_embedding = nn.Embedding(num_users, embed_size, device=device)
        self.movie_embedding = nn.Embedding(num_movies, embed_size, device=device)
        self.gender_embedding = nn.Embedding(num_genders, embed_size, device=device)
        self.occupation_embedding = nn.Embedding(num_occupations, embed_size, device=device)
        self.age_embedding = nn.Embedding(num_age_bins, embed_size, device=device)
        self.genre_embedding = nn.Embedding(num_genres, embed_size, device=device)

        # Linear layer for wide features
        self.linear_layer = nn.Linear(embed_size * 6, 1, device=device)

        # Deep component
        self.deep_stack = nn.Sequential(
            nn.Linear(embed_size * 6, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 64, device=device),
            nn.ReLU(),
            nn.Linear(64, 32, device=device),
            nn.ReLU()
        )
        self.output = nn.Linear(32 + 1, 1, device=device)  # Combine FM and Deep parts

    def forward(self, X_trains):
        import idpb; ipdb.set_trace()
        # Embedding lookup for each feature
        user_embed = self.user_embedding(user)
        movie_embed = self.movie_embedding(movie)
        gender_embed = self.gender_embedding(gender)
        occupation_embed = self.occupation_embedding(occupation)
        age_embed = self.age_embedding(age)
        genre_embed = self.genre_embedding(genres)

        # Concatenate all embeddings
        x = torch.cat([user_embed, movie_embed, gender_embed, occupation_embed, age_embed, genre_embed], dim=1)

        # Linear and FM part
        fm_out = self.linear_layer(x)

        # Deep part
        deep_out = self.deep_stack(x)

        # Final output
        out = self.output(torch.cat((deep_out, fm_out), dim=1))
        return out.squeeze(1)

# Training and evaluation loop
def train_deepfm(model:DeepFM, X_train, y_train, X_valid, y_valid, X_test, y_test, learning_rate=1e-3, weight_decay=0.01, num_epochs=10):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, valid_losses = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        import ipdb; ipdb.set_trace()  
        y_pred_train = model(**X_train)
        loss_train = loss_fn(y_pred_train, y_train.float())
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_pred_valid = model(**X_valid)
            loss_valid = loss_fn(y_pred_valid, y_valid.float())

        train_losses.append(loss_train.item())
        valid_losses.append(loss_valid.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Valid Loss: {loss_valid.item():.4f}")

    # Testing
    model.eval()
    with torch.no_grad():
        y_pred_test = model(**X_test)
        test_mse, test_mae = calculate_metrics(y_test.float(), y_pred_test)
        print(f"Test Results - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

    return model

# Main execution
X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_process_data()
import ipdb; ipdb.set_trace()

deepfm_model = DeepFM(
    num_users=X_train['user id'].nunique(),
    num_movies=X_train['movie id'].nunique(),
    num_genders=2,
    num_occupations=X_train['occupation'].nunique(),
    num_age_bins=7,  # Based on age intervals
    num_genres=19,  # Number of genres in MovieLens dataset
    embed_size=16
)

trained_model = train_deepfm(deepfm_model, X_train, y_train, X_valid, y_valid, X_test, y_test, num_epochs=20)