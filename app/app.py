from flask import Flask, render_template, request
import torch
import pandas as pd

# Import your recommendation model and necessary preprocessing functions
from model import recommend_books_based_on_mood
from model import fetch_book_details
from model import store_ratings_in_model

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
        #return render_template("rating.html", book_details=book_details)
        # Render the recommendations on the frontend
        if(len(book_details) == 10):
            return render_template("rating.html", book_details=book_details, user_id=user_id)
        else:
            return render_template("recommendations.html", book_details=book_details)
    else:
        # If the method is not POST, handle it accordingly (e.g., return an error response)
        return "Method not allowed", 405
    
@app.route("/submit-ratings-endpoint", methods=["POST"])
def submit_ratings():
    if request.method == "POST":
        # Assuming each book has an ID associated with it
        book_ratings = {}
        user_id = int(request.form.get("user_id"))  # Retrieve user_id from the form data
        
        # Iterate through form data to get book ratings
        for key, value in request.form.items():
            if key.startswith("rating-"):
                book_id = key.replace("rating-", "")
                book_ratings[book_id] = 6 - int(value)
        
        # Call a function in model.py to store the ratings
        print("Book ratings are:")
        print(book_ratings.items())
        store_ratings_in_model(book_ratings, user_id)
        
        # Redirect or render a success message
        return render_template("index.html")
    else:
        return "Method not allowed", 405
if __name__ == "__main__":
    app.run(debug=True)
