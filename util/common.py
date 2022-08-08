import os

import pandas as pd

try:
    from pydub import AudioSegment
except ImportError:
    pass


# pylint: disable=line-too-long
USECOLS = ['pos', 'track_name', 'track_id', 'weeks', 'top10', 'peak', 'peak_x', 'peak_streams', 'totals', 'preview_url', 'release_date']

class ChartCol:
    POS = 'pos'
    TRACK_NAME = 'track_name'
    TRACK_ID = 'track_id'
    WEEKS = 'weeks'
    TOP10 = 'top10'
    PEAK = 'peak'
    PEAK_X = 'peak_x'
    PEAK_STREAMS = 'peak_streams'
    TOTALS = 'totals'
    PREVIEW_URL = 'preview_url'
    RELEASE_DATE = 'release_date'

def load_chart(chart_path: str):
    df: pd.DataFrame = pd.read_csv(chart_path, usecols=USECOLS)

    df = df[df[ChartCol.PREVIEW_URL].notna()]
    return df

    # df[Chart.RELEASE_DATE] = [parse_date(date_string) for date_string in df[Chart.RELEASE_DATE]]
    # df = df[df[Chart.RELEASE_DATE] != None]

    # df = df[df[Chart.RELEASE_DATE].notna()]

def mp3_to_wav(mp3_file_path: str, output_dir: str = None):
    # song = AudioSegment.from_mp3(mp3_file_path, )
    song = AudioSegment.from_file(mp3_file_path, 'mp3', frame_rate='16000')
    filename = os.path.basename(os.path.splitext(mp3_file_path)[0])
    output_dir = output_dir or os.path.join(mp3_file_path, '..')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{filename}.wav')
    song.export(output_path, format='wav')
    return
