import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data preparation
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

    # Encode categorical features
    label_encoder = LabelEncoder()
    dataset['gender'] = (dataset['gender'] == 'M').astype(int)
    dataset['occupation'] = label_encoder.fit_transform(dataset['occupation'])

    # Normalize ratings
    dataset['rating'] = dataset['rating'] / dataset['rating'].max()

    # Drop irrelevant columns
    dataset.drop(['zip code', 'movie title', 'release date', 'video release date', 'IMDb URL', 'timestamp'], axis=1, inplace=True)

    # Split into train, validation, and test sets
    train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    return train_data, valid_data, test_data

class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = torch.tensor(row.drop('rating').values, dtype=torch.float32, device=device)
        target = torch.tensor(row['rating'], dtype=torch.float32, device=device)
        return features, target

# DeepFM model as per D2L style
class DeepFM(nn.Module):
    def __init__(self, num_features, embed_size):
        super(DeepFM, self).__init__()
        self.num_features = num_features
        self.embed_size = embed_size

        # Embedding layer for features
        self.embedding = nn.Embedding(num_features, embed_size)

        # Linear part
        self.linear = nn.Linear(num_features, 1, bias=True)

        # Deep part
        self.deep = nn.Sequential(
            nn.Linear(num_features * embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(32 + 1, 1)

    def forward(self, x):
        # Linear part
        linear_part = self.linear(x)

        # Embedding part
        embed_part = self.embedding(x.long())
        embed_part = embed_part.view(embed_part.size(0), -1)  # Flatten embeddings for deep part

        # Deep part
        deep_part = self.deep(embed_part)

        # Combine linear and deep parts
        out = self.final_layer(torch.cat([linear_part, deep_part], dim=1))
        return out.squeeze(1)

# Training loop
def train_deepfm(model, train_loader, valid_loader, num_epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for features, target in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

# Main execution
train_data, valid_data, test_data = load_and_process_data()
train_dataset = MovieLensDataset(train_data)
valid_dataset = MovieLensDataset(valid_data)
test_dataset = MovieLensDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

num_features = train_data.shape[1] - 1  # Excluding the target column
deepfm_model = DeepFM(num_features=num_features, embed_size=16).to(device)

train_deepfm(deepfm_model, train_loader, valid_loader, num_epochs=20)