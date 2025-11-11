# utils/recommend.py
import pandas as pd
import numpy as np

# --- Define emotional zones per mood ---
MOOD_CHARACTERISTICS = {
    "Happy":     {"valence": (0.6, 1.0), "energy": (0.6, 1.0)},
    "Sad":       {"valence": (0.0, 0.4), "energy": (0.0, 0.5)},
    "Energetic": {"valence": (0.4, 0.8), "energy": (0.7, 1.0)},
    "Calm":      {"valence": (0.4, 0.7), "energy": (0.2, 0.6)},
}

def recommend_songs(df: pd.DataFrame, mood: str, language: str = None, n: int = 8) -> pd.DataFrame:
    """
    Recommend songs using valence & energy proximity.
    Includes smart fallback when the initial filter yields few or no results.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    mood = (mood or "Calm").capitalize()
    if mood not in MOOD_CHARACTERISTICS:
        mood = "Calm"

    v_lo, v_hi = MOOD_CHARACTERISTICS[mood]["valence"]
    e_lo, e_hi = MOOD_CHARACTERISTICS[mood]["energy"]

    # --- Primary filter ---
    filtered = df[
        df["valence"].between(v_lo, v_hi, inclusive="both") &
        df["energy"].between(e_lo, e_hi, inclusive="both")
    ]

    # --- Language filter ---
    if language:
        filtered_lang = filtered[filtered["language"].str.lower() == language.lower()]
        # if language filter yields results, use it; else fallback to broader mood-based
        if not filtered_lang.empty:
            filtered = filtered_lang

    # --- Fallback 1: widen valence/energy band if too few songs ---
    if len(filtered) < n // 2:
        v_lo, v_hi = max(0, v_lo - 0.1), min(1, v_hi + 0.1)
        e_lo, e_hi = max(0, e_lo - 0.1), min(1, e_hi + 0.1)
        filtered = df[
            df["valence"].between(v_lo, v_hi, inclusive="both") &
            df["energy"].between(e_lo, e_hi, inclusive="both")
        ]

        if language:
            fallback_lang = filtered[filtered["language"].str.lower() == language.lower()]
            if not fallback_lang.empty:
                filtered = fallback_lang

    # --- Fallback 2: full dataset if still empty ---
    if filtered.empty:
        filtered = df.copy()

    # --- Compute "distance" to the ideal emotional center ---
    v_center = (v_lo + v_hi) / 2
    e_center = (e_lo + e_hi) / 2
    filtered = filtered.copy()
    filtered["score"] = (filtered["valence"] - v_center) ** 2 + (filtered["energy"] - e_center) ** 2

    # --- Sort and return top-N unique songs ---
    top = (
        filtered
        .sort_values("score", ascending=True)
        .drop_duplicates(subset=["name", "artist"])
        .head(n)
    )

    return top[["name", "artist", "language", "mood", "valence", "energy"]]


# --- Standalone test ---
if __name__ == "__main__":
    df = pd.read_csv("data/final_master_song_dataset.csv")
    print(recommend_songs(df, mood="Happy", language="Hindi", n=5))
