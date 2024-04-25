from flask import Flask, render_template, request
import torch
import pandas as pd

# Import your recommendation model and necessary preprocessing functions
from model import recommend_books
from model import get_book_names

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
        mood = request.form.get("mood")
        book_details = recommend_books("26", mood)  # "26" should be dynamically set or authenticated user ID
        book_details = get_book_names(book_details)

        # Debug output
        print("***************************************")
        print("Books details received from recommendation system:", book_details)
        print("***************************************")

        # Render the recommendations on the frontend
        return render_template("recommendations.html", book_details=book_details)


if __name__ == "__main__":
    app.run(debug=True)