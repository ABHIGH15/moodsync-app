# utils/youtube_spotify.py
import urllib.parse

def safe_str(value):
    """Ensure string safety for URL building."""
    return str(value or "").strip()

def build_youtube_link(song_name: str, artist: str) -> str:
    """
    Builds a fast, direct YouTube search URL — no API or subprocess.
    Adds 'official music video' keywords for higher match relevance.
    """
    query = f"{safe_str(song_name)} {safe_str(artist)} official music video"
    encoded = urllib.parse.quote_plus(query)
    return f"https://www.youtube.com/results?search_query={encoded}"

def build_spotify_link(song_name: str, artist: str) -> str:
    """
    Builds a Spotify search URL for the given song and artist.
    """
    query = f"{safe_str(song_name)} {safe_str(artist)}"
    encoded = urllib.parse.quote_plus(query)
    return f"https://open.spotify.com/search/{encoded}"

def attach_links(recommendations: list) -> list:
    """
    Adds YouTube & Spotify search links to each recommended song.
    Render-safe: no network calls, subprocess, or heavy dependencies.
    """
    if not isinstance(recommendations, list) or len(recommendations) == 0:
        return []

    for r in recommendations:
        try:
            name = r.get("name", "")
            artist = r.get("artist", "")
            r["youtube"] = build_youtube_link(name, artist)
            r["spotify"] = build_spotify_link(name, artist)
        except Exception as e:
            print(f"⚠️ Link generation error: {e}")
            r["youtube"] = "#"
            r["spotify"] = "#"

    return recommendations
