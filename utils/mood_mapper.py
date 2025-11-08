# utils/mood_mapper.py
import pandas as pd

def infer_mood(row):
    valence, energy = row['valence'], row['energy']
    if valence >= 0.6 and energy >= 0.6:
        return 'Happy'
    elif valence < 0.4 and energy < 0.5:
        return 'Sad'
    elif energy > 0.75 and valence < 0.6:
        return 'Energetic'
    else:
        return 'Calm'

def add_mood_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'mood' not in df.columns:
        df['mood'] = df.apply(infer_mood, axis=1)
    df['mood'] = df['mood'].astype(str).str.capitalize()
    return df
