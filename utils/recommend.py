# utils/recommend.py
import pandas as pd

# Define target emotional regions for each mood
MOOD_CHARACTERISTICS = {
    "Happy":     {"valence": (0.6, 1.0), "energy": (0.6, 1.0)},
    "Sad":       {"valence": (0.0, 0.4), "energy": (0.0, 0.5)},
    "Energetic": {"valence": (0.4, 0.8), "energy": (0.7, 1.0)},
    "Calm":      {"valence": (0.4, 0.7), "energy": (0.2, 0.6)},
}

def recommend_songs(df: pd.DataFrame, mood: str, language: str = None, n: int = 8) -> pd.DataFrame:
    """
    Recommend songs based on mood + language using valence & energy proximity.
    """
    mood = (mood or "Calm").capitalize()
    if mood not in MOOD_CHARACTERISTICS:
        mood = "Calm"

    v_lo, v_hi = MOOD_CHARACTERISTICS[mood]["valence"]
    e_lo, e_hi = MOOD_CHARACTERISTICS[mood]["energy"]

    # Base filtering
    base = df[(df['valence'].between(v_lo, v_hi)) & (df['energy'].between(e_lo, e_hi))]

    # Language filtering (if given)
    if language:
        base = base[base['language'].str.lower() == language.lower()] or base

    if base.empty:
        base = df.copy()

    # Distance to the "ideal" emotion center
    v_center = (v_lo + v_hi) / 2
    e_center = (e_lo + e_hi) / 2
    base = base.copy()
    base['score'] = (base['valence'] - v_center)**2 + (base['energy'] - e_center)**2

    # Top N closest matches
    top = base.sort_values('score').head(n)
    return top[['name','artist','language','mood','valence','energy']]

if __name__ == "__main__":
    df = pd.read_csv("data/final_master_song_dataset.csv")
    print(recommend_songs(df, mood="Happy", language="English", n=5))

