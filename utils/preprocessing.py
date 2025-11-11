# utils/preprocessing.py
import pandas as pd
import logging

# --- Expected Columns for MoodSync Dataset ---
REQUIRED_COLS = [
    "name", "artist", "danceability", "acousticness", "energy",
    "instrumentalness", "liveness", "valence", "loudness",
    "speechiness", "tempo", "key", "language", "mood"
]

def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads and cleans the main song dataset for MoodSync.
    Ensures column presence, fixes casing, removes duplicates, and normalizes fields.
    """
    logging.info(f"📂 Loading dataset from: {path}")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logging.error(f"❌ Dataset not found at {path}")
        raise
    except Exception as e:
        logging.error(f"⚠️ Error reading dataset: {e}")
        raise

    # Normalize column names (trim + lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Drop duplicates and empty rows
    before_rows = len(df)
    df.drop_duplicates(subset=["name", "artist"], inplace=True)
    df.dropna(subset=["mood", "language"], inplace=True)
    after_rows = len(df)

    logging.info(f"🧹 Cleaned dataset: removed {before_rows - after_rows} duplicates/invalid rows")

    # Normalize text casing
    df["language"] = df["language"].astype(str).str.strip().str.capitalize()
    df["mood"] = df["mood"].astype(str).str.strip().str.capitalize()

    # Fill optional numeric columns safely (if they exist)
    for col in ["valence", "energy", "danceability"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.5)

    logging.info(f"✅ Dataset loaded successfully: {df.shape[0]} songs, {df.shape[1]} columns")
    logging.info(f"🎧 Available moods: {sorted(df['mood'].unique())}")
    logging.info(f"🌐 Languages: {sorted(df['language'].unique())}")

    return df


# --- Local Test Run ---
if __name__ == "__main__":
    data = load_dataset("data/final_master_song_dataset.csv")
    print("✅ Dataset loaded successfully:", data.shape)
    print("🎧 Moods:", data["mood"].unique())
    print("🌐 Languages:", data["language"].unique())
