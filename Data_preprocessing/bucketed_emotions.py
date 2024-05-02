import pandas as pd
import ast
from collections import OrderedDict

# Define the mapping of emotions to buckets
bucket_mapping = {
    'Melancholic': ['sad', 'powerless', 'demoralized', 'cheated', 'melancholy', 'depressed', 'mournful', 'hopeless', 'lugubrious', 'dolorous', 'funereal', 'disheartened', 'sorrowful', 'mumpish', 'devastated', 'despairing', 'grief-stricken', 'despondent', 'afflicted', 'discontented', 'disconsolate', 'crestfallen', 'heavyhearted', 'heartbroken', 'crushed', 'miserable', 'unhappy', 'woeful', 'woebegone', 'lamentable', 'regretful', 'remorse', 'downhearted', 'melancholic', 'despairing'],
    'Joyful': ['happy', 'cheerful', 'elated', 'ecstatic', 'glad', 'joyful', 'merry', 'jovial', 'jubilant', 'blissful', 'delighted', 'pleased', 'contented', 'thrilled', 'exultant', 'fantastic', 'fabulous', 'exuberant', 'spirited', 'high-spirited', 'euphoric', 'overjoyed', 'rapturous', 'gleeful', 'sunny', 'buoyant', 'lighthearted', 'enthusiastic', 'beatific', 'serene', 'tranquil', 'convivial', 'jaunty', 'vivacious', 'terrific', 'self-satisfied', 'peaceful'],
    'Motivational': ['motivated', 'encouraged', 'inspired', 'determined', 'enterprising', 'intent', 'resolute', 'dedicated', 'ardent', 'fervent', 'committed', 'ambitious', 'courageous', 'gallant', 'audacious', 'dauntless', 'heroic', 'bold', 'brave', 'stout-hearted', 'daring', 'fearless'],
    'Fearful': ['fearful', 'anxious', 'terrified', 'frightened', 'alarmed', 'scared', 'apprehensive', 'nervous', 'worried', 'tense', 'panic', 'frightened', 'timid', 'hesitant', 'insecure', 'doubtful', 'wary', 'cautious', 'distrustful', 'skeptical', 'leery', 'suspicious', 'afraid', 'petrified', 'horror-struck', 'aghast', 'horrified', 'panic-stricken', 'terrorized', 'tremulous', 'fainthearted', 'trembling', 'quaking'],
    'Romantic': ['romantic', 'lustful', 'passionate', 'infatuated', 'enamored', 'attracted', 'loving', 'affectionate', 'ardent', 'desirous', 'enamored', 'horny', 'turned on']
}

# Function to assign emotions to buckets and sort the result
def assign_and_sort_bucket(emotion_list):
    emotion_dict = ast.literal_eval(emotion_list)
    bucketed_emotions = {
        'Melancholic': 0,
        'Joyful': 0,
        'Motivational': 0,
        'Fearful': 0,
        'Romantic': 0
    }
    for emotion, count in emotion_dict:
        emotion = emotion.strip()
        for bucket, emotions in bucket_mapping.items():
            if emotion in emotions:
                bucketed_emotions[bucket] += count
    # Sort the dictionary by values in descending order
    sorted_bucketed_emotions = OrderedDict(sorted(bucketed_emotions.items(), key=lambda x: x[1], reverse=True))
    return sorted_bucketed_emotions

# Function to sum dictionaries
def sum_dictionaries(dict1, dict2):
    summed_dict = {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}
    sorted_summed = OrderedDict(sorted(summed_dict.items(), key=lambda x: x[1], reverse=True))
    return sorted_summed


def max_key_from_dict(bucket_dict):
    if bucket_dict:
        return max(bucket_dict, key=bucket_dict.get)  # Return the key corresponding to the maximum value
    return None  # Return None if the dictionary is empty or not provided

# Read the CSV file
df = pd.read_csv('/content/all_ratings.csv')

# Apply the function to each row in the 'Aggregated Emotions' column
df['Sorted Buckets'] = df['Aggregated Emotions'].apply(assign_and_sort_bucket)
df['Sorted Buckets desc'] = df['Aggregated Des Emotions'].apply(assign_and_sort_bucket)

# Sum the dictionaries in 'Sorted Buckets' and 'Sorted Buckets desc'
df['Total Buckets'] = df.apply(lambda row: sum_dictionaries(row['Sorted Buckets'], row['Sorted Buckets desc']), axis=1)

df['Max Mood'] = df['Total Buckets'].apply(max_key_from_dict)

# Show the updated DataFrame
print(df[['Aggregated Emotions', 'Sorted Buckets', 'Sorted Buckets desc', 'Total Buckets']])

# Optionally, save the modified DataFrame to a new CSV file
df.to_csv('bucketed_emotions.csv', index=False)