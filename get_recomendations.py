# get_recommendations.py
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler

# Load your library and your taste profile
df = pd.read_csv('my_library_analyzed.csv')
with open('my_taste_profile.json', 'r') as f:
    taste_profile = json.load(f)

# Convert profile back to pandas Series for easy math
taste_numerical = pd.Series(taste_profile['numerical_profile'])
taste_genres = taste_profile['genre_profile']

# --- Calculate a "Match Score" for every song ---

# 1. Score for numerical features (how close is this song to the average?)
# We use a simple distance metric. Lower is better.
numerical_cols = ['year', 'avg_rms', 'avg_spectral_centroid']
# Normalize the data to make features comparable
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
taste_numerical_scaled = scaler.transform([taste_numerical])[0]

df['numerical_score'] = (df[numerical_cols] - taste_numerical_scaled).abs().sum(axis=1)

# 2. Score for genre (how common is this song's genre in your library?)
# We'll use the genre frequency as a bonus score.
df['genre_score'] = df['genre'].map(taste_genres).fillna(0)

# 3. Combine the scores!
# We want to *minimize* the numerical distance and *maximize* the genre score.
# A simple way is to subtract the genre score from the numerical score.
# We can add a 'weight' to make genre more or less important.
GENRE_WEIGHT = 0.5 # You can tweak this value!
df['final_match_score'] = df['numerical_score'] - (df['genre_score'] * GENRE_WEIGHT)


# --- Get the Recommendations ---
# Sort by the best (lowest) final score
recommendations = df.sort_values('final_match_score', ascending=True)

# --- Display the Top 50 ---
print("--- Your Personalized 'Radio' Playlist ---")
print("(Top 50 songs most representative of your library's taste)\n")
top_50 = recommendations.head(50)
for i, row in top_50.iterrows():
    print(f"{row['artist']} - {row['title']} ({row['genre']}, {int(row['year'])})")

# Optional: Save the full sorted list to a CSV for your media player
recommendations.to_csv('my_library_ranked.csv', index=False)
print("\nFull ranked list saved to my_library_ranked.csv")