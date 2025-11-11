# utils/youtube_spotify.py
import urllib.parse

def build_youtube_link(song_name: str, artist: str) -> str:
    """
    Creates a direct YouTube search link instead of downloading metadata.
    """
    query = urllib.parse.quote(f"{song_name} {artist}")
    return f"https://www.youtube.com/results?search_query={query}"

def build_spotify_link(song_name: str, artist: str) -> str:
    """
    Creates a Spotify search link.
    """
    query = urllib.parse.quote(f"{song_name} {artist}")
    return f"https://open.spotify.com/search/{query}"

def attach_links(recommendations: list) -> list:
    """
    Adds YouTube & Spotify links to each song in the list.
    """
    if not recommendations or not isinstance(recommendations, list):
        return []

    for r in recommendations:
        try:
            r["youtube"] = build_youtube_link(r["name"], r["artist"])
            r["spotify"] = build_spotify_link(r["name"], r["artist"])
        except Exception as e:
            print(f"⚠️ Link generation error: {e}")
            r["youtube"] = "#"
            r["spotify"] = "#"
    return recommendations
