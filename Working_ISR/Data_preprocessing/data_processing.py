import pandas as pd
from textblob import TextBlob
import re
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tkinter as tk
from tkinter import messagebox
import tqdm
import nltk

# Check if the punkt tokenizer is downloaded, if not, download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('stopwords')

# Method which returns the clean text given as input
def cleanTxt(text):
    print(text)
    # Convert the text to lower case
    text = text.lower()
    # Remove punctuations: !"#$%&'()*+,-.:;<=>?@[\]^_`{|}~/
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Replace ... with ' '
    text = re.sub(r'\.\.\.', ' ', text)
    # Remove URLs
    text = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', ' ', text)
    # Remove emails
    text = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', " ", text)
    # Remove duplicated letters e.g. "I loooooveeee this book"
    text = re.sub("(.)\\1{2,}", "\\1", text)
    # Remove quotes from the book
    text = re.sub(r'\"[A-Za-z0-9 ,.?!%&()@$-_:;\\]+\"', ' ', text)
    # Remove *notes* (e.g. **spoilers**)
    text = re.sub(r'\*[A-Za-z0-9 ,.?!%&()@$-_:;\\]+\*', ' ', text)
    # Remove notes written in ()
    text = re.sub(r'\([A-Za-z0-9 ,.?!%&()@$-_:;\\]+\)', ' ', text)
    # Remove new lines and tabs
    text = re.sub(r'\n +', '', text)
    text = re.sub(r'\n+', '', text)
    text = re.sub(r'\t+', '', text)

    tokenized_words = word_tokenize(text, "english")
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)
    text = ''
    for word in final_words:
        text = text + word + ' '
    return text


# Method which returns the subjectivity computed using TextBlob library method
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Method which returns the polarity computed using TextBlob library method
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getEmotion(text, returnType):
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        # Wrap the file object with tqdm to display a progress bar
        for line in tqdm.tqdm(file, desc='Processing lines'):
            clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
            if ':' in clear_line:
                word, emotion = clear_line.split(':')
                # Here, you should compare normalized versions of both word and text_word
                for text_word in text.split(" "):
                    if re.match(r'^' + re.escape(text_word) + r'$', word, re.IGNORECASE):
                        emotion_list.append(emotion)

    w = Counter(emotion_list)
    subset = w.most_common(30)

    if returnType == "list":
        return subset
    elif returnType == "counter":
        return w
    else:
        return emotion_list

# Method to compute the negative, neutral and positive analysis
def getAnalysisPositiveNegativeNeutral(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


# Main method for processing the data
def process_data():
    df_reviews = pd.read_csv("/Users/priyankaaskani/Downloads/ISR_project/merged_reviews.csv", index_col=[0])
    #df_books = pd.read_csv("data/database_books.csv", encoding='latin-1', index_col=[0])
    df_reviews = df_reviews.dropna(subset=['Description'])
    df_reviews['Reviews'] = df_reviews['Reviews'].apply(cleanTxt)
    df_reviews['Description'] = df_reviews['Description'].apply(cleanTxt)
    print(df_reviews['Reviews'].head(5))

    # Compute the values for Subjectivity, Polarity and Polarity Classification columns
    df_reviews['Subjectivity'] = df_reviews['Reviews'].apply(getSubjectivity)
    df_reviews['Polarity'] = df_reviews['Reviews'].apply(getPolarity)
    df_reviews['Polarity Classification'] = df_reviews['Polarity'].apply(getAnalysisPositiveNegativeNeutral)

    # # Print in the console the overview of the new columns
    print(df_reviews[['Reviews', 'Subjectivity', 'Polarity', 'Polarity Classification']])

    # Get percentage of positive comments
    pcomments = df_reviews[df_reviews['Polarity Classification'] == 'Positive']['Reviews']
    print(round((pcomments.shape[0] / df_reviews.shape[0]) * 100, 1))

    # # Get percentage of negative comments
    ncomments = df_reviews[df_reviews['Polarity Classification'] == 'Negative']['Reviews']
    print(round((ncomments.shape[0] / df_reviews.shape[0]) * 100, 1))

    # Retrieve the emotions from the Review Content and store them in the column Emotions
    df_reviews['Emotions'] = df_reviews['Reviews'].apply(getEmotion, returnType="list")
    df_reviews['Emotions_description'] = df_reviews['Description'].apply(getEmotion, returnType="list")
    # Store the new/edited reviews in a new file (different than the original database_reviews)
    df_reviews.to_csv("/Users/priyankaaskani/Downloads/ISR_project/merge_reviews_emo.csv")


    print("FINISHED DATA PROCESSING")

    # A pop up window appears to announce the input datasets were processed.
    tk.messagebox.showinfo('Info', 'Process input datasets Completed')
