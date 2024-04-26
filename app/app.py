from flask import Flask, render_template, request
import torch
import pandas as pd

# Import your recommendation model and necessary preprocessing functions
from model import recommend_books_based_on_mood
from model import fetch_book_details

app = Flask(__name__)

# Load necessary data and models
# Load pre-trained recommendation model
# Load other necessary data (e.g., book metadata)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    if request.method == "POST":
        # Get user's selected mood from the form
        mood = request.form.get("mood")
        user_id = int(request.form.get("userId"))

        # Call your recommendation function with the user's mood
        top_3_book_isbns = recommend_books_based_on_mood(mood, user_id)
        print("In app.py")
        book_details = fetch_book_details(top_3_book_isbns)

        for book_detail in book_details:
            # Assuming book_detail is a dictionary containing book information
            # Add the image URL to each book detail
            # You need to replace 'image_url_key' with the appropriate key containing the image URL in your book details dictionary
            book_detail['image_url'] = get_image_url_from_isbn(book_detail['isbn'])  # Replace get_image_url_from_isbn with your function to get image URL from ISBN
            book_detail['url'] = get_book_url_from_isbn(book_detail['isbn'])

        # Render the recommendations on the frontend
        return render_template("recommendations.html", book_details=book_details)

def get_image_url_from_isbn(isbn):
    # Load book data from CSV file
    book_data = pd.read_csv('data/all_books.csv')

    # This function should return the image URL associated with the given ISBN
    # You need to implement this function based on how you store and retrieve image URLs in your data
    # For example, if you have a DataFrame containing book metadata where you can look up image URLs based on ISBN, you can implement it like this:
    # Assuming you have a DataFrame called book_data containing book metadata with columns: 'ISBN' and 'Image-URL-S'
    image_url = book_data.loc[book_data['ISBN'] == isbn, 'Image-URL-S'].values[0]
    return image_url

def get_book_url_from_isbn(isbn):
    # Load book data from CSV file
    book_data = pd.read_csv('data/all_books.csv')

    # This function should return the image URL associated with the given ISBN
    # You need to implement this function based on how you store and retrieve image URLs in your data
    # For example, if you have a DataFrame containing book metadata where you can look up image URLs based on ISBN, you can implement it like this:
    # Assuming you have a DataFrame called book_data containing book metadata with columns: 'ISBN' and 'Image-URL-S'
    book_url = book_data.loc[book_data['ISBN'] == isbn, 'URL'].values[0]
    return book_url

if __name__ == "__main__":
    app.run(debug=True)
