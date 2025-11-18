# analyze_library.py
import os
import pandas as pd
from mutagen.easyid3 import EasyID3
import librosa
import warnings

# --- CONFIGURATION ---
# Replace with the path to your music library
MUSIC_LIBRARY_PATH = '/path/to/your/library/'
OUTPUT_CSV = 'my_library_analyzed.csv'

# Suppress librosa's warnings for cleaner output
warnings.filterwarnings("ignore")

print("Starting library analysis... This will take a while. Go watch an epic movie or read a book.")

music_data = []
processed_count = 0

for root, dirs, files in os.walk(MUSIC_LIBRARY_PATH):
    for file in files:
        if file.endswith(".mp3"):
            file_path = os.path.join(root, file)
            try:
                # 1. Read Metadata (the easy part)
                audio = EasyID3(file_path)
                artist = audio.get('artist', ['Unknown Artist'])[0]
                title = audio.get('title', ['Unknown Title'])[0]
                album = audio.get('album', ['Unknown Album'])[0]
                genre = audio.get('genre', ['Unknown Genre'])[0]
                year = audio.get('date', ['0'])[0]
                try:
                    year = int(year)
                except (ValueError, TypeError):
                    year = 0

                # 2. Light Audio Analysis (the lazy part)
                # We only calculate two fast features: RMS Loudness and Spectral Centroid
                y, sr = librosa.load(file_path, sr=None, mono=True)
                
                # RMS (Root Mean Square) - a good proxy for perceived loudness/energy
                rms = librosa.feature.rms(y=y)
                avg_rms = float(rms.mean())

                # Spectral Centroid - indicates the "brightness" of the sound
                # (e.g., high for cymbals, low for bass)
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                avg_spectral_centroid = float(spectral_centroids.mean())

                # 3. Store the data
                music_data.append({
                    'file_path': file_path,
                    'artist': artist,
                    'title': title,
                    'album': album,
                    'genre': genre,
                    'year': year,
                    'avg_rms': avg_rms,
                    'avg_spectral_centroid': avg_spectral_centroid
                })
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")

            except Exception as e:
                print(f"Could not process {file_path}: {e}")

# Save everything to a CSV file
df = pd.DataFrame(music_data)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nAnalysis complete! Processed {processed_count} files.")
print(f"Data saved to {OUTPUT_CSV}")