import os

import dotenv
import requests
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

# handler = logging.FileHandler('log.log', 'a', encoding='utf-8')
# logger = logging.getLogger()
# logger.addHandler(handler)


class SpotifyService:

    def __init__(self, dotenv_path: str):

        assert os.path.exists(dotenv_path)
        dotenv.load_dotenv(dotenv_path)
        self.sp = Spotify(auth_manager=SpotifyClientCredentials())
        return

    def get_track_preview_url(self, track_id: str):

        track = self.sp.track(track_id)

        return track['preview_url']


def save_preview(preview_url: str, output_path: str):

    if not output_path.endswith('.mp3'):
        output_path += '.mp3'
    
    with requests.get(preview_url) as res:
        with open(output_path, 'wb') as fout:
            fout.write(res.content)
    return
