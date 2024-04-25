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

        # Call your recommendation function with the user's mood
        top_3_book_isbns = recommend_books_based_on_mood(mood)

        book_details = fetch_book_details(top_3_book_isbns)

        # Render the recommendations on the frontend
        return render_template("recommendations.html", book_details=book_details)

if __name__ == "__main__":
    app.run(debug=True)
