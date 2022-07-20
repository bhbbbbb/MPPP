import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from .common import load_track, InvalidDateException

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log_.log', 'a', encoding='utf8'))
logger.setLevel(logging.DEBUG)


# pylint: disable=line-too-long
USECOLS = ['pos', 'track_name', 'track_id', 'weeks', 'top10', 'peak', 'peak_x', 'peak_streams', 'totals', 'preview_url', 'release_date']

class Chart:
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

    df = df[df[Chart.PREVIEW_URL].notna()]
    return df

    # df[Chart.RELEASE_DATE] = [parse_date(date_string) for date_string in df[Chart.RELEASE_DATE]]
    # df = df[df[Chart.RELEASE_DATE] != None]

    # df = df[df[Chart.RELEASE_DATE].notna()]


def make_sets(chart: pd.DataFrame, region: str, stream_root: str,output_dir: str = 'sets'):

    def try_load_track(track_id: str):
        try:
            load_track(track_id, region, stream_root)
            return track_id
        except InvalidDateException:
            print(f'invalid: {track_id}')
            return None
    
    track_ids = []

    for track_id in chart[Chart.TRACK_ID]:
        if try_load_track(track_id):
            track_ids.append(track_id)

    os.makedirs(output_dir, exist_ok=True)

    all_set = pd.DataFrame({'track_id': track_ids})

    train_set, remain = train_test_split(all_set, test_size=.3, random_state=0xAAA)

    valid_set, test_set = train_test_split(remain, test_size=.5, random_state=0XAAA)

    print('n_samples(train, valid, test): ', len(train_set), len(valid_set), len(test_set))
    train_set: pd.DataFrame
    valid_set: pd.DataFrame
    test_set: pd.DataFrame
    
    all_set.to_csv(os.path.join(output_dir, 'all.csv'))
    train_set.to_csv(os.path.join(output_dir, 'train.csv'))
    valid_set.to_csv(os.path.join(output_dir, 'valid.csv'))
    test_set.to_csv(os.path.join(output_dir, 'test.csv'))
    return

def get_metrices_from_track_history():
    pass

if __name__ == '__main__':
    # pylint: disable=line-too-long
    CHART_PATH = 'D:\\Documents\\PROgram\\song-prediction\\data-collector\\data\\chart\\jp_weekly_totals.csv'
    STREAM_ROOT = 'D:\\Documents\\PROgram\\song-prediction\\data-collector\\data\\track\\streams\\'
