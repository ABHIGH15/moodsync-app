# utils/preprocessing.py
import pandas as pd

REQUIRED_COLS = [
    'name','artist','danceability','acousticness','energy','instrumentalness',
    'liveness','valence','loudness','speechiness','tempo','key','language'
]

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    # minimal cleanup
    df['language'] = df['language'].astype(str).str.strip().str.capitalize()
    return df
