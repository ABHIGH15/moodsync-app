# utils/recommend.py
import pandas as pd

MOOD_CHARACTERISTICS = {
    "Happy":     {"valence": (0.6, 1.0), "energy": (0.6, 1.0)},
    "Sad":       {"valence": (0.0, 0.4), "energy": (0.0, 0.5)},
    "Energetic": {"valence": (0.4, 0.8), "energy": (0.7, 1.0)},
    "Calm":      {"valence": (0.4, 0.7), "energy": (0.2, 0.6)},
}

def recommend_songs(df: pd.DataFrame, mood: str, language: str, n=8) -> pd.DataFrame:
    mood = (mood or "Calm").capitalize()
    if mood not in MOOD_CHARACTERISTICS:
        mood = "Calm"

    v_lo, v_hi = MOOD_CHARACTERISTICS[mood]["valence"]
    e_lo, e_hi = MOOD_CHARACTERISTICS[mood]["energy"]

    base = df[(df['valence'].between(v_lo, v_hi)) & (df['energy'].between(e_lo, e_hi))]
    if language:
        lang_mask = base['language'].str.lower() == language.lower()
        preferred = base[lang_mask]
        if not preferred.empty:
            base = preferred

    if base.empty:
        base = df.copy()

    # distance to ideal (valence, energy) center
    v_center = (v_lo + v_hi) / 2
    e_center = (e_lo + e_hi) / 2
    base = base.copy()
    base['score'] = (base['valence'] - v_center)**2 + (base['energy'] - e_center)**2

    top = base.sort_values('score').head(n)
    return top[['name','artist','language','mood','valence','energy']]
