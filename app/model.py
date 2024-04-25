import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dot, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Custom function to handle unseen labels
def transform_with_fallback(encoder, labels, default='unknown'):
    if not hasattr(encoder, 'classes_'):
        raise ValueError("This LabelEncoder instance is not fitted yet.")
    unseen_labels = [label for label in labels if label not in encoder.classes_]
    if unseen_labels:
        if default not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, default)
        return encoder.transform([default if label in unseen_labels else label for label in labels])
    else:
        return encoder.transform(labels)

# Load the dataset
merged_books_ratings = pd.read_csv('data/books_with_rats_moods.csv')
print(merged_books_ratings.columns)

# Encoding user IDs, book IDs, and Moods
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()
mood_encoder = LabelEncoder()

merged_books_ratings['User-ID'] = merged_books_ratings['User-ID'].astype(str)

merged_books_ratings['user_id_encoded'] = user_encoder.fit_transform(merged_books_ratings['User-ID'])
merged_books_ratings['book_id_encoded'] = book_encoder.fit_transform(merged_books_ratings['Book'])
#mood_encoder.fit(list(merged_books_ratings['Max Mood']) + ['unknown'])  # Including 'unknown' label
mood_encoder.fit([m.lower() for m in merged_books_ratings['Max Mood']] + ['unknown'])
#merged_books_ratings['mood_encoded'] = transform_with_fallback(mood_encoder, merged_books_ratings['Max Mood'])
merged_books_ratings['mood_encoded'] = transform_with_fallback(mood_encoder, [m.lower() for m in merged_books_ratings['Max Mood']], default='unknown')


print("Mood classes after fitting:", mood_encoder.classes_)
print("User ID classes in encoder:", user_encoder.classes_)
print("User ID '26' in encoder classes:", '26' in user_encoder.classes_)
#print("Unique moods in dataset:", merged_books_ratings['Max Mood'].unique())



# Normalize ratings
merged_books_ratings['Book-Rating'] = merged_books_ratings['Book-Rating'].apply(lambda x: (x - 1) / 9)

# Split the data
train, test = train_test_split(merged_books_ratings, test_size=0.2, random_state=42)

# Model architecture for collaborative filtering
def build_collaborative_filtering_model(num_users, num_books, embedding_size=15):
    user_input = Input(shape=(1,))
    book_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, embedding_size, embeddings_regularizer=l2(1e-6))(user_input)
    book_embedding = Embedding(num_books, embedding_size, embeddings_regularizer=l2(1e-6))(book_input)
    user_vec = Flatten()(user_embedding)
    book_vec = Flatten()(book_embedding)
    dot_product = Dot(axes=1)([user_vec, book_vec])
    user_bias = Flatten()(Embedding(num_users, 1)(user_input))
    book_bias = Flatten()(Embedding(num_books, 1)(book_input))
    sum = Add()([dot_product, user_bias, book_bias])
    model = Model([user_input, book_input], sum)
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    return model

# Build and train the model
model = build_collaborative_filtering_model(len(user_encoder.classes_), len(book_encoder.classes_))
model.fit([train['user_id_encoded'], train['book_id_encoded']], train['Book-Rating'], batch_size=64, epochs=5, validation_split=0.1)

# Function to recommend books based on user similarity and mood
def recommend_books(user_id, mood, top_n=5):
    try:
        user_id_str = str(user_id)  # Convert to string to match the encoding
        mood_str = mood.lower()     # Convert to lowercase to match the encoding

        print("Looking for user:", user_id_str)
        print("Looking for mood:", mood_str)

        # Check if the user and mood are in the encoder classes
        if user_id_str not in user_encoder.classes_:
            return f"User ID '{user_id_str}' not found in dataset."
        if mood_str not in mood_encoder.classes_:
            return f"Mood '{mood_str}' not found in dataset."

        user_idx = user_encoder.transform([user_id_str])
        mood_idx = mood_encoder.transform([mood_str])
        print(f"User index: {user_idx}")
        print(f"Mood index: {mood_idx}")

        # Get the valid books for the mood index
        valid_books = merged_books_ratings[merged_books_ratings['mood_encoded'] == mood_idx[0]]['book_id_encoded'].unique()
        #print(f"Valid books indices: {valid_books}")
        print(f"Number of valid books for mood '{mood_str}': {len(valid_books)}")

        if len(valid_books) == 0:
            return f"No books found for mood '{mood_str}'."

        # Predict the ratings for the valid books
        predictions = model.predict([np.array([user_idx[0]] * len(valid_books)), valid_books])
        #print(f"Predictions: {predictions}")

        top_books_idx = predictions.flatten().argsort()[-top_n:][::-1]
        recommended_books = book_encoder.inverse_transform(top_books_idx)

        return recommended_books
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_book_names(book_list):
    # Assume book_list is a list of titles as strings
    if isinstance(book_list, list):
        # Convert list of titles into a list of dictionaries
        return [{'title': book, 'author': 'Author Name'} for book in book_list]
    else:
        return [{'title': book_list, 'author': 'Author Name'}]  # Handle error messages similarly


# Test the recommendation function
test_user_id = "26"  # Use a valid user ID for testing
test_mood = "Melancholic"  # Example mood
print("Recommended Books:", recommend_books(test_user_id, test_mood))

# Evaluate the model
eval_loss = model.evaluate([test['user_id_encoded'], test['book_id_encoded']], test['Book-Rating'])
print(f'Evaluation Loss: {eval_loss}')
