import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def recommend_books_based_on_mood(mood, user_id):
    ratings=pd.read_csv('data/all_ratings.csv', low_memory=False)
    ratings_df=pd.read_csv('data/all_ratings.csv',low_memory=False)
    # Initialize LabelEncoders for user and item IDs
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    # Fit and transform user and item IDs
    ratings['user_id'] = user_encoder.fit_transform(ratings['User-ID'])
    ratings['item_id'] = item_encoder.fit_transform(ratings['ISBN'])


    # Fit and transform user and item IDs
    ratings_df['user_id'] = user_encoder.fit_transform(ratings_df['User-ID'])
    ratings_df['item_id'] = item_encoder.fit_transform(ratings_df['ISBN'])
    ratings_train = ratings[0:36890]
    ratings_test = ratings[36890:].reset_index(drop=True)
    class RatingDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {
                'user_id': self.data['user_id'][idx],
                'book_id': self.data['item_id'][idx],
                #'book_id': self.data['ISBN'][idx],
                'rating': self.data['Book-Rating'][idx]
            }
        


    # Instantiate train and test datasets
    train_dataset = RatingDataset(ratings_train)
    test_dataset = RatingDataset(ratings_test)

    # Create train and test loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    class GMF(nn.Module):
        def __init__(self, num_users, num_items, embedding_size):
            super(GMF, self).__init__()
            self.relu = nn.ReLU()
            self.user_embedding = nn.Embedding(num_users, embedding_size)
            self.item_embedding = nn.Embedding(num_items, embedding_size)
            self.fc = nn.Linear(embedding_size, 32)
            self.output_layer = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.2)

        def forward(self, user_ids, item_ids):
            user_embed = self.user_embedding(user_ids)
            item_embed = self.item_embedding(item_ids)
            element_product = user_embed * item_embed
            x = self.fc(element_product)
            x = self.relu(x)
            x = self.dropout(x)
            output = self.output_layer(x)
            output = torch.sigmoid(output)  # Ensure output is between 0 and 1
            return output.view(-1)
        
    class MLP(nn.Module):
        def __init__(self, num_users, num_items, embedding_size, hidden_layers=[64, 32]):
            super(MLP, self).__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_size)
            self.item_embedding = nn.Embedding(num_items, embedding_size)
            layers = []
            input_size = embedding_size * 2
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                input_size = hidden_size
            layers.append(nn.Linear(hidden_layers[-1], 1))
            self.layers = nn.Sequential(*layers)

        def forward(self, user_ids, item_ids):
            user_embed = self.user_embedding(user_ids)
            item_embed = self.item_embedding(item_ids)
            concat_embed = torch.cat((user_embed, item_embed), dim=1)
            output = self.layers(concat_embed)
            output = torch.sigmoid(output)  # Ensure output is between 0 and 1
            return output.view(-1)

    class NCF(nn.Module):
        def __init__(self, gmf_model, mlp_model):
            super(NCF, self).__init__()
            self.gmf = gmf_model
            self.mlp = mlp_model

        def forward(self, user_ids, item_ids):
            gmf_output = self.gmf(user_ids, item_ids)
            mlp_output = self.mlp(user_ids, item_ids)
            combined_output = (gmf_output + mlp_output) / 2
            return combined_output
        
    num_users = len(ratings['User-ID'].unique())
    num_items = len(ratings['ISBN'].unique())
    embedding_size = 64
    hidden_layers = [128, 64, 32]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # Initialize GMF model
    gmf_model = GMF(num_users, num_items, embedding_size).to(device)

    # Initialize MLP model
    mlp_model = MLP(num_users, num_items, embedding_size, hidden_layers).to(device)

    #Loss criterion for GMF and MLP models
    models_criterion = nn.MSELoss()

    # Optimizer for GMF model
    gmf_optimizer = optim.Adam(gmf_model.parameters(), lr=0.001)

    # Optimizer for MLP model
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

    def train_gmf(model, dataloader, criterion, optimizer, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0.0
            i=0
            total_diff=0
            total_len=0
            for batch in dataloader:
                user_ids = batch['user_id'].to(device)
                item_ids = batch['book_id'].to(device)
                ratings = batch['rating'].to(device)

                optimizer.zero_grad()
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, (ratings.float()/10) )
                loss.backward()
                optimizer.step()

                if (i % 1000 == 0):
                    actual_ratings = 10*predictions
                    diff = torch.abs( actual_ratings - ratings ).sum().item()
                    print(f'Batch [{i+1}/{len(dataloader)}], Loss: {loss.item()}, Avg. Diff: { (diff/len(ratings)) }')

                i = i + 1

                total_loss += loss.item()

            print(f'GMF Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')

    def train_mlp(model, dataloader, criterion, optimizer, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0.0
            i=0
            for batch in dataloader:
                user_ids = batch['user_id'].to(device)
                item_ids = batch['book_id'].to(device)
                ratings = batch['rating'].to(device)

                optimizer.zero_grad()
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, (ratings.float()/10))
                loss.backward()
                optimizer.step()

                if (i % 1000 == 0):
                    actual_ratings = 10*predictions
                    diff = torch.abs( actual_ratings - ratings ).sum().item()
                    print(f'Batch [{i+1}/{len(dataloader)}], Loss: {loss.item()}, Avg. Diff: { (diff/len(ratings)) }')

                i = i + 1

                total_loss += loss.item()

            print(f'MLP Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')

    num_epochs = 5

    print("Training GMF...")
    train_gmf(gmf_model, train_loader, models_criterion, gmf_optimizer, num_epochs)

    print("Training MLP...")
    train_mlp(mlp_model, train_loader, models_criterion, mlp_optimizer, num_epochs)

    model = NCF(gmf_model, mlp_model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        i=0
        for batch in train_loader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['book_id'].to(device)
            ratings = batch['rating'].to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, (ratings.float()/10))
            loss.backward()
            optimizer.step()

            if (i % 1000 == 0):
                actual_ratings = 10*predictions
                rmse = math.sqrt(torch.square(actual_ratings-ratings).sum().item()/len(ratings))
                diff = torch.abs( actual_ratings - ratings ).sum().item()
                print(f'Batch [{i+1}/{len(train_loader)}], Loss: {loss.item()}, Avg. Diff: { (diff/len(ratings)) }, RMSE: {rmse}')


            i = i + 1


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    model.eval()

    total_loss = 0
    total_diff = 0
    total_examples = 0
    total_squared_error = 0
    with torch.no_grad():
        for batch in test_loader:
            user_ids = batch['user_id'].to(device)
            item_ids = batch['book_id'].to(device)
            ratings = batch['rating'].to(device)

            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, (ratings.float() / 10))

            actual_ratings = 10 * predictions
            diff = torch.abs(actual_ratings - ratings).sum().item()

            total_loss += loss.item() * len(ratings)
            total_diff += diff
            total_examples += len(ratings)
            total_squared_error += torch.square(actual_ratings-ratings).sum().item()

    avg_loss = total_loss / total_examples
    avg_diff = total_diff / total_examples
    rmse = math.sqrt(total_squared_error / total_examples)

    print('Evalution Measures:')
    print(f'Evaluation Loss: {avg_loss}, Average Difference: {avg_diff}, RMSE: {rmse}')

    book_ratings = ratings_df.groupby('item_id')['Book-Rating'].mean().reset_index()
    book_ratings = book_ratings.sort_values(by='Book-Rating', ascending=False)
    top_64_books = book_ratings.head(64)
    print(top_64_books)
    #user_id = 276704
    user_id_tensor = torch.LongTensor([user_id] * 64).to(device)

    top_64_books = top_64_books['item_id'].tolist()
    item_ids_tensor = torch.LongTensor(top_64_books).to(device)

    print("User ID Tensor:", user_id_tensor)
    print("Top 10 Books Tensor:", item_ids_tensor)

    # Clip user indices to valid range
    user_id_tensor = torch.clip(user_id_tensor, 0, num_users - 1)

    # Clip item indices to valid range
    item_ids_tensor = torch.clip(item_ids_tensor, 0, num_items - 1)

    print(user_id_tensor)
    print(item_ids_tensor)
    predictions = model(user_id_tensor, item_ids_tensor)

    indexed_predictions = [(idx, pred) for idx, pred in enumerate(predictions)]

    # Sort the indexed predictions by the prediction values in descending order
    sorted_predictions = sorted(indexed_predictions, key=lambda x: x[1], reverse=True)

    # Get the top 3 unique ISBNs
    top_3_unique_isbns = []
    seen_isbns = set()  # Keep track of seen ISBNs to ensure uniqueness
    for idx, pred in sorted_predictions:
        isbn = ratings_df['ISBN'].iloc[idx]
        if isbn not in seen_isbns:
            top_3_unique_isbns.append(isbn)
            seen_isbns.add(isbn)
        if len(top_3_unique_isbns) == 3:
            break

    print("Top 3 Unique ISBNs:", top_3_unique_isbns)

    return top_3_unique_isbns


# Assuming you have a DataFrame named book_data with columns ['ISBN', 'Title', 'Author']
book_data = pd.read_csv('data/all_books.csv')  # Load your book data from a CSV file or database

def fetch_book_details(book_isbns):
    book_details = []
    for isbn in book_isbns:
        # Find book details based on ISBN
        book_info = book_data[book_data['ISBN'] == isbn]
        print("here")
        print(book_info)
        print("here")
        if not book_info.empty:
            title = book_info.iloc[0]['Book']
            print("title is: ", title)
            author = book_info.iloc[0]['Author']
            print("author is ", author)
            book_details.append({'isbn': isbn, 'title': title, 'author': author})
    return book_details