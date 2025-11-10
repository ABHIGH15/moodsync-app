# utils/youtube_spotify.py
from youtubesearchpython import VideosSearch
import urllib.parse

def get_youtube_link(song_name: str, artist: str = "") -> str:
    """
    Search YouTube for the song + artist and return the first video link.
    """
    query = f"{song_name} {artist}".strip()
    try:
        results = VideosSearch(query, limit=1).result()
        if results and 'result' in results and len(results['result']) > 0:
            return results['result'][0]['link']
        else:
            return f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
    except Exception as e:
        print(f"[YouTube API Error] {e}")
        return f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"


def get_spotify_link(song_name: str, artist: str = "") -> str:
    """
    Generate a direct Spotify search link (no API key required).
    """
    query = f"{song_name} {artist}".strip()
    encoded_query = urllib.parse.quote(query)
    return f"https://open.spotify.com/search/{encoded_query}"


def add_streaming_links(df):
    """
    Add YouTube and Spotify links to each recommended song.
    """
    df = df.copy()
    df["youtube_link"] = df.apply(lambda x: get_youtube_link(x["name"], x["artist"]), axis=1)
    df["spotify_link"] = df.apply(lambda x: get_spotify_link(x["name"], x["artist"]), axis=1)
    return df


# ✅ Quick local test
if __name__ == "__main__":
    import pandas as pd
    data = {
        "name": ["Perfect", "Blinding Lights"],
        "artist": ["Ed Sheeran", "The Weeknd"]
    }
    df = pd.DataFrame(data)
    out = add_streaming_links(df)
    print(out[["name", "youtube_link", "spotify_link"]])

