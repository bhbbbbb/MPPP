import os
from threading import Thread
from queue import Queue

import pandas as pd
from tqdm import tqdm

from .chart_parser import ChartParser
from .track_parser import TrackParser
from .spotify_api import SpotifyService, save_preview

def crawl_chart(
    country: str,
    dotenv_path: str,
    output_dir: str = 'chart',
    overwrite: bool = False
):

    output_filepath = os.path.join(output_dir, f'{country.lower()}_weekly_totals.csv')
    
    if not overwrite and os.path.exists(output_filepath):
        return output_filepath
    
    html = ChartParser.fetch(country)

    df = ChartParser.parse(html)

    sps = SpotifyService(dotenv_path)

    tracks = (sps.get_track(track_id) for track_id in tqdm(df['track_id']))

    tem = ((track.preview_url, track.release_date) for track in tracks)

    df['preview_url'], df['release_date'] = list(zip(*tem))

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_filepath)
    return output_filepath

def crawl_track(track_id: str, output_dir: str = 'track', overwrite: bool = False):

    pos_output_dir = os.path.join(output_dir, 'pos')
    stream_output_dir = os.path.join(output_dir, 'streams')

    pos_output_filepath = os.path.join(pos_output_dir, f'{track_id}.csv')
    stream_output_filepath = os.path.join(stream_output_dir, f'{track_id}.csv')

    if (
        not overwrite
        and os.path.exists(pos_output_filepath)
        and os.path.exists(stream_output_filepath)
    ):
        return

    os.makedirs(stream_output_dir, exist_ok=True)
    os.makedirs(pos_output_dir, exist_ok=True)
    

    html = TrackParser.fetch(track_id)

    pos_df, stream_df = TrackParser.parse(html)

    pos_df.to_csv(pos_output_filepath)
    stream_df.to_csv(stream_output_filepath)
    return

def crawl_tracks_in_chart(
    chart_csv_path: str,
    limit: int = None,
    offset: int = 0,
    output_dir: str = 'track',
    overwrite: bool = False,
):

    chart_df = pd.read_csv(chart_csv_path, usecols=['track_id', 'preview_url'])

    preview_dir = os.path.join(output_dir, 'preview')
    os.makedirs(preview_dir, exist_ok=True)

    print(chart_df)

    pbar = tqdm(
        zip(chart_df['track_id'][offset:limit], chart_df['preview_url'][offset:limit]),
        total=len(chart_df[offset:limit])
    )

    END = 'end'
    q1 = Queue(maxsize=10)
    def mp3_downloader():
        while True:
            args = q1.get()
            if args == END:
                return
            save_preview(*args)
            q1.task_done()
    
    q2 = Queue(maxsize=10)
    def track_crawler():
        while True:
            args = q2.get()
            if args == END:
                return
            crawl_track(*args)
            q2.task_done()

    th1 = Thread(target=mp3_downloader, daemon=True)
    th1.start()
    th2 = Thread(target=track_crawler, daemon=True)
    th2.start()

    for track_id, preview_url in pbar:

        mp3_path = os.path.join(preview_dir, f'{track_id}.mp3')

        if pd.isna(preview_url) or os.path.exists(mp3_path):
            continue

        # crawl_track(track_id, output_dir, overwrite)
        q2.put((track_id, output_dir, overwrite))
        # save_preview(preview_url, mp3_path)
        q1.put((preview_url, mp3_path))

    q2.join()
    q1.join()
    q1.put(END)
    q2.put(END)
    th2.join()
    th1.join()
    print('done')
    return
