import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scipy
import math
import sklearn
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
import pandas as pd
import math
import numpy as np
import random
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

def get_train_test_split(dataframe, min_interactions=3, test_size=0.30, random_state=42):
    def apply_log_smoothing(value):
        return math.log(1 + value, 2)

    # Group by ISBN and User-ID, count interactions, and summarize by user
    interaction_summary = dataframe.groupby(['ISBN', 'User-ID']).size()
    user_interaction_totals = interaction_summary.groupby('User-ID').size()
    print(f'Total number of users: {len(user_interaction_totals)}')

    # Filter users with at least 'min_interactions' interactions
    qualified_users = user_interaction_totals[user_interaction_totals >= min_interactions].reset_index()[['User-ID']]
    print(f'Users with at least {min_interactions} interactions: {len(qualified_users)}')

    # Filter the dataset to include only interactions from qualified users
    qualified_interactions = dataframe.merge(qualified_users, on='User-ID', how='right')
    print(f'Total interactions: {len(dataframe)}')
    print(f'Interactions from qualified users: {len(qualified_interactions)}')

    # Apply smoothing to the sum of book ratings and reset the index
    smoothed_interactions = qualified_interactions.groupby(['ISBN', 'User-ID'])['Book-Rating'].sum().apply(apply_log_smoothing).reset_index()
    print(f'Unique user/item interactions: {len(smoothed_interactions)}')

    # Split data into training and testing sets
    train_data, test_data = train_test_split(smoothed_interactions,
                                             stratify=smoothed_interactions['User-ID'],
                                             test_size=test_size,
                                             random_state=random_state)
    print(f'Interactions on Train set: {len(train_data)}')
    print(f'Interactions on Test set: {len(test_data)}')

    return train_data, test_data, smoothed_interactions

"""#SVD"""

def matrix_factorization_predictions(train_df, num_factors=15):
    """Performs matrix factorization using SVD on the user-item ratings matrix from training data.

    Args:
        train_df (DataFrame): Training data containing user, item, and ratings.
        num_factors (int): Number of latent factors to use in the matrix factorization.

    Returns:
        DataFrame: A DataFrame with the predicted ratings for all users and items.
    """
    # Create pivot table
    users_items_pivot_matrix_df = train_df.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

    # Convert the pivot table to a sparse matrix format
    users_items_pivot_matrix = csr_matrix(users_items_pivot_matrix_df.values)

    # Perform matrix factorization using SVD
    U, sigma, Vt = svds(users_items_pivot_matrix, k=num_factors)
    sigma = np.diag(sigma)

    # Predict ratings
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    # Normalize the ratings to be between 0 and 10
    min_val = np.min(all_user_predicted_ratings)
    max_val = np.max(all_user_predicted_ratings)
    all_user_predicted_ratings_norm = ((all_user_predicted_ratings - min_val) /
                                       (max_val - min_val)) * 10

    # Convert the matrix back to a DataFrame
    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns, index=users_items_pivot_matrix_df.index).transpose()

    return cf_preds_df

"""#User based Collaborative Filtering using Matrix factorization"""

# Assuming your CFRecommender class has a method recommend_items that can return ratings predictions
# First, extend your CFRecommender class to include a method to predict ratings for a given user and item

class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df):
        self.cf_predictions_df = cf_predictions_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['ISBN'].isin(items_to_ignore)].sort_values('recStrength', ascending=False).head(topn)
        return recommendations_df

    def predict_rating(self, user_id, item_id):
        if user_id in self.cf_predictions_df.columns and item_id in self.cf_predictions_df.index:
            return self.cf_predictions_df.loc[item_id, user_id]
        else:
            return np.nan  # Return NaN for user/item combinations not in the matrix

class ModelRecommender:

    def __init__(self, interactions_full_indexed_df,interactions_test_indexed_df, interactions_train_indexed_df, ratings_df_unique ):
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.ratings_df_unique = ratings_df_unique

    def get_items_interacted(UserID, interactions_df):
      interacted_items = interactions_df.loc[UserID]['ISBN']
      return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    # Function for getting the set of items which a user has not interacted with
    def get_not_interacted_items_sample(self, UserID, sample_size, seed=42):
        interacted_items = self.get_items_interacted(UserID, self.interactions_full_indexed_df)
        all_items = set(self.ratings_df['ISBN'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

#
    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index


    # Fetch top 10 highly rated books
    def get_top_k_popular_books(self, topk=5):
      books_grouped = self.interactions_test_indexed_df.groupby('ISBN').size().reset_index(name='count')
      
      top_books = books_grouped.sort_values('count', ascending=False).head(topk)
      
      
      # Since top_books_details might have multiple entries for the same book, we'll drop duplicates
      top_books = top_books.drop_duplicates(subset=['ISBN'])
      
      top_books = top_books[['ISBN']]
      return top_books


    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, person_id, mood):

        # Getting the items in test set
        interacted_values_testset = self.interactions_test_indexed_df.loc[person_id]

        if type(interacted_values_testset['ISBN']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['ISBN'])])

        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from the model for a given user
        #person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id, interactions_train_indexed_df),topn=10000000000)
        person_recs_df = model.recommend_items(person_id, items_to_ignore=[],topn=10000000000)
        print(person_recs_df)
        updated_person_recs_df = person_recs_df.merge(self.ratings_df_unique[['ISBN', 'Max Mood', 'Book']], on='ISBN', how='left')
        print(updated_person_recs_df.head(10))
        updated_person_recs_df = updated_person_recs_df[updated_person_recs_df['Max Mood'].str.contains(mood, na=False)]
        print('Recommendation for User-ID = ',person_id)
        return updated_person_recs_df.head(5)

        # Function to evaluate the performance of model at overall level
    def recommend_book(self, model ,userid, mood):

        person_metrics = self.evaluate_model_for_user(model, userid, mood)
        return person_metrics

#model_recommender = ModelRecommender()

"""#BUILD model"""

def build_model():
  ratings_df = pd.read_csv("data/baseline_ratinsg.csv")
  ratings_df.head()
  ratings_df.rename(columns={'user_id':'User-ID','isbn':'ISBN','book_rating':'Book-Rating'},inplace=True)
  ratings_df_unique = ratings_df.drop_duplicates(subset='ISBN')
  train_df, test_df, interactions_full_df = get_train_test_split(ratings_df)
  print(f'Interactions on Train set: %d' % len(train_df))
  print(f'Interactions on Test set: %d' % len(test_df))
  cf_preds_df = matrix_factorization_predictions(train_df)
  cf_preds_df.head()
  interactions_full_indexed_df = interactions_full_df.set_index('User-ID')
  interactions_train_indexed_df = train_df.set_index('User-ID')
  interactions_test_indexed_df = test_df.set_index('User-ID')
  cf_recommender_model = CFRecommender(cf_preds_df)
  model_recommender = ModelRecommender(interactions_full_indexed_df,interactions_test_indexed_df, interactions_train_indexed_df, ratings_df_unique)
  return model_recommender, cf_recommender_model, test_df, train_df

def recommend_books_based_on_mood(mood, user_id):
  model_recommender, cf_recommender_model, test_df, train_df = build_model()
  
  if user_id in model_recommender.interactions_test_indexed_df.index:
    interacted_values_testset = model_recommender.interactions_test_indexed_df.loc[user_id]
    interaction_count = interacted_values_testset.shape[0] if model_recommender.interactions_test_indexed_df.loc[user_id].ndim > 1 else 1

    if interaction_count < 3:
      print(f"Less than 3 interactions for user {user_id}. Get Top 10 highly rated books")
      top_k_books_isbn = model_recommender.get_top_k_popular_books(10)

      list_top_k_books_isbn = list(top_k_books_isbn['ISBN'])
      list_top_k_books_isbn = [number.zfill(10) for number in list_top_k_books_isbn]
      print(list_top_k_books_isbn)
      return list_top_k_books_isbn
  else:
    print(f"No interactions found for user {user_id}.")
    top_k_books_isbn = model_recommender.get_top_k_popular_books(10)

    list_top_k_books_isbn = list(top_k_books_isbn['ISBN'])
    list_top_k_books_isbn = [number.zfill(10) for number in list_top_k_books_isbn]
    print(list_top_k_books_isbn)
    return list_top_k_books_isbn

  ret_updated_person_recs_df = model_recommender.recommend_book(cf_recommender_model,user_id,mood)
  list_ret_updated_person_recs_df = list(ret_updated_person_recs_df['ISBN'])
  list_ret_updated_person_recs_df = [number.zfill(10) for number in list_ret_updated_person_recs_df]
  print(list_ret_updated_person_recs_df)
  return list_ret_updated_person_recs_df


#print((recommend_books_based_on_mood('Fearful', 21576)))

"""#Evaluation

# Now, predict ratings for all user-item pairs in the test set
test_users = test_df['User-ID']
test_items = test_df['ISBN']
predicted_ratings = [cf_recommender_model.predict_rating(user, item) for user, item in zip(test_users, test_items)]

# Add these predictions back to the test dataframe
test_df['predicted_rating'] = predicted_ratings

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_df['Book-Rating'], test_df['predicted_rating'].fillna(0)))
print(f"RMSE: {rmse}")"""

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
            image_url = book_info.iloc[0]['Image-URL-S']
            print("image_url is ", image_url)
            url = book_info.iloc[0]['URL']
            print("url is ", url)
            book_details.append({'isbn': isbn, 'title': title, 'author': author, 'image_url' : image_url, 'url': url})
    return book_details



import csv

'''def store_ratings_in_model(ratings, user_id, filename='/Users/prathyushapolepalli/Documents/ISR/AuraRead/data/baseline_ratinsg.csv'):
    # Define the fieldnames for the CSV file
    fieldnames = ['User-ID', 'ISBN', 'Book-Rating']
    
    # Write ratings to CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write each rating entry
        for isbn, rating in ratings.items():
            writer.writerow({'User-ID': user_id, 'ISBN': isbn, 'Book-Rating': rating*2})'''


def store_ratings_in_model(ratings, user_id, filename='/Users/prathyushapolepalli/Documents/ISR/AuraRead/data/baseline_ratinsg.csv'):
    # Define the fieldnames for the CSV file
    fieldnames = ['Unnamed: 0', 'Book', 'Author', 'Description', 'Genres', 'Year of Publication', 'Publisher_x', 'URL', 'Aggregated Emotions', 'Aggregated Des Emotions', 'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher_y', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'User-ID', 'Book-Rating', 'Sorted Buckets', 'Sorted Buckets desc', 'Total Buckets', 'Max Mood']
    
    # Write ratings to CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write each rating entry
        for isbn, rating in ratings.items():
            row = {key: '' for key in fieldnames}  # Initialize row with empty values
            row['User-ID'] = user_id
            row['ISBN'] = isbn
            row['Book-Rating'] = rating*2
            writer.writerow(row)



