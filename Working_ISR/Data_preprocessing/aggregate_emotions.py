import pandas as pd
from collections import Counter
from ast import literal_eval

# Load the original dataset, this should be the path to your CSV file
# Since I can't access your file directly, you'll need to specify the path to your CSV
df_original = pd.read_csv('/Users/priyankaaskani/Downloads/ISR_project/merge_reviews_emo.csv')

# Make sure the 'Emotions' column in 'df_original' is converted to lists of tuples
df_original['Emotions'] = df_original['Emotions'].apply(literal_eval)
df_original['Emotions_description'] = df_original['Emotions_description'].apply(literal_eval)


# Group by 'Book Id' and aggregate emotions
def aggregate_emotions(group):
    emotion_counter = Counter()
    for emotions in group['Emotions']:
        emotion_counter.update(dict(emotions))
    return sorted(emotion_counter.items(), key=lambda x: x[1], reverse=True)

# Group by 'Book Id' and aggregate emotions
def aggregate_des_emotions(group):
    emotion_counter = Counter()
    for emotions in group['Emotions_description']:
        emotion_counter.update(dict(emotions))
    return sorted(emotion_counter.items(), key=lambda x: x[1], reverse=True)

aggregated_emotions = df_original.groupby('Book').apply(aggregate_emotions).reset_index(name='Aggregated Emotions')
aggregated_des_emotions = df_original.groupby('Book').apply(aggregate_des_emotions).reset_index(name='Aggregated Des Emotions')

# Now, exclude 'Reviews' from the original DataFrame, and also remove duplicate 'Book Id's if they exist
df_final = df_original.drop(columns=['Reviews', 'Emotions', 'Emotions_description', 'Subjectivity', 'Polarity', 'Polarity Classification']).drop_duplicates('Book')

# Join the aggregated emotions with the original DataFrame without 'Reviews'
df_final = df_final.merge(aggregated_emotions, on='Book')
df_final = df_final.merge(aggregated_des_emotions, on='Book')
# Save the final DataFrame to a new CSV file
df_final.to_csv('/Users/priyankaaskani/Downloads/ISR_project/merged_emotions_all.csv', index=False)
