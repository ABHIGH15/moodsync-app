%%writefile utils/youtube_spotify.py
from youtubesearchpython import VideosSearch
from urllib.parse import quote_plus

def build_spotify_search_url(name: str, artist: str) -> str:
    query = quote_plus(f"{name} {artist}")
    return f"https://open.spotify.com/search/{query}"

def build_youtube_link(name: str, artist: str) -> str:
    query = f"{name} {artist}"
    try:
        search = VideosSearch(query, limit=1)
        return search.result()['result'][0]['link']
    except Exception:
        return f"https://www.youtube.com/results?search_query={quote_plus(query)}"

def attach_links(rows):
    out = []
    for r in rows:
        yt = build_youtube_link(r['name'], r['artist'])
        sp = build_spotify_search_url(r['name'], r['artist'])
        out.append({**r, 'youtube': yt, 'spotify': sp})
    return out
