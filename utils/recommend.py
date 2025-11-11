# utils/recommend.py
import pandas as pd
import random

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
    Adds randomness for variety while keeping cluster coherence.
    """
    if df.empty:
        return pd.DataFrame()

    mood = (mood or "Calm").capitalize()
    if mood not in MOOD_CHARACTERISTICS:
        mood = "Calm"

    v_lo, v_hi = MOOD_CHARACTERISTICS[mood]["valence"]
    e_lo, e_hi = MOOD_CHARACTERISTICS[mood]["energy"]

    # 1️⃣ Language filtering (first)
    if language:
        base = df[df['language'].str.lower() == language.lower()]
        if base.empty:
            base = df.copy()
    else:
        base = df.copy()

    # 2️⃣ Select a random "offset" within mood bounds to simulate sub-clusters
    offset_val = random.uniform(-0.05, 0.05)
    offset_eng = random.uniform(-0.05, 0.05)
    v_center = ((v_lo + v_hi) / 2) + offset_val
    e_center = ((e_lo + e_hi) / 2) + offset_eng

    # 3️⃣ Score distance from this dynamic center
    base = base.copy()
    base["score"] = (base["valence"] - v_center) ** 2 + (base["energy"] - e_center) ** 2

    # 4️⃣ Random sampling after sorting to keep freshness
    top_candidates = base.sort_values("score").head(50)  # take 50 close ones
    sampled = top_candidates.sample(n=min(n, len(top_candidates)), replace=False, random_state=None)

    return sampled[['name', 'artist', 'language', 'mood', 'valence', 'energy']]
