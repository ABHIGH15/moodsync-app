# utils/youtube_spotify.py
import subprocess
from urllib.parse import quote_plus

def build_spotify_search_url(name: str, artist: str) -> str:
    query = quote_plus(f"{name} {artist}")
    return f"https://open.spotify.com/search/{query}"

def build_youtube_link(name: str, artist: str) -> str:
    """Get top YouTube search result URL using yt-dlp"""
    query = f"{name} {artist}"
    try:
        # Run yt-dlp search quietly
        result = subprocess.run(
            ["yt-dlp", f"ytsearch1:{query}", "--get-id", "--no-warnings"],
            capture_output=True,
            text=True,
            timeout=10
        )
        video_id = result.stdout.strip()
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        print("⚠️ YouTube search error:", e)
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"

def attach_links(rows):
    out = []
    for r in rows:
        yt = build_youtube_link(r['name'], r['artist'])
        sp = build_spotify_search_url(r['name'], r['artist'])
        out.append({**r, 'youtube': yt, 'spotify': sp})
    return out
