import os

import dotenv
import requests
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

# handler = logging.FileHandler('log.log', 'a', encoding='utf-8')
# logger = logging.getLogger()
# logger.addHandler(handler)

class Track:

    def __init__(self, naive_track: dict):
        self._content = naive_track
        return

    @property
    def preview_url(self):
        return self._content['preview_url']
    
    @property
    def release_date(self):
        return self._content['album']['release_date']


class SpotifyService:

    _cached_id: str
    _cached_track: dict

    def __init__(self, dotenv_path: str):

        assert os.path.exists(dotenv_path)
        dotenv.load_dotenv(dotenv_path)
        self.sp = Spotify(auth_manager=SpotifyClientCredentials())
        self._cached_id = self.cached_track = None
        return

    def get_track(self, track_id: str):
        return Track(self.sp.track(track_id))

def save_preview(preview_url: str, output_path: str):

    if not output_path.endswith('.mp3'):
        output_path += '.mp3'
    
    with requests.get(preview_url) as res:
        with open(output_path, 'wb') as fout:
            fout.write(res.content)
    return
