# utils/preprocessing.py
import pandas as pd

REQUIRED_COLS = [
    'name','artist','danceability','acousticness','energy','instrumentalness',
    'liveness','valence','loudness','speechiness','tempo','key','language','mood'
]

def load_dataset(path: str) -> pd.DataFrame:
    """Load and clean the song dataset for MoodSync app"""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure all required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Basic cleaning
    df.drop_duplicates(subset=['name','artist'], inplace=True)
    df.dropna(subset=['mood','language'], inplace=True)

    # Normalize casing
    df['language'] = df['language'].astype(str).str.strip().str.capitalize()
    df['mood'] = df['mood'].astype(str).str.strip().str.capitalize()

    return df

if __name__ == "__main__":
    data = load_dataset("data/final_master_song_dataset.csv")
    print("✅ Dataset loaded successfully:", data.shape)
    print("🎧 Available moods:", data['mood'].unique())
    print("🌐 Languages:", data['language'].unique())
