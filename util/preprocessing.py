from datetime import date
import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from .common import ChartCol, load_chart
from .track import Track, InvalidDateException

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log_.log', 'a', encoding='utf8'))
logger.setLevel(logging.DEBUG)



def make_sets(
    chart_path: str,
    *,
    region: str,
    stream_root: str,
    now_date: date,
    output_dir: str = 'sets',
):

    chart = load_chart(chart_path)

    def try_load_track(track_id: str):
        try:
            track = Track(
                track_id,
                use_region=region,
                stream_root=stream_root,
                now_date=now_date,
            )
            return track.is_valid()
        except InvalidDateException:
            print(f'invalid: {track_id}')
            return None
    
    track_ids = []

    for track_id in chart[ChartCol.TRACK_ID]:
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

def combine_sets(
    primary_set_dir: str,
    secondary_set_dir: str,
    primary_region: str,
    secondary_region: str,
    output_dir: str,
):

    os.makedirs(output_dir, exist_ok=True)

    __set_name__ = ['all.csv', 'train.csv', 'valid.csv', 'test.csv']

    for set_name in __set_name__:

        primary_path = os.path.join(primary_set_dir, set_name)
        secondary_path = os.path.join(secondary_set_dir, set_name)

        assert os.path.exists(primary_path)
        assert os.path.exists(secondary_path)

        output_path = os.path.join(output_dir, set_name)
        _combine_set(primary_path, secondary_path, primary_region, secondary_region, output_path)
    
    return


def _combine_set(
    primary_set_path: str,
    secondary_set_path: str,
    primary_region: str,
    secondary_region: str,
    output_path: str,
):
    
    primary = pd.read_csv(primary_set_path)['track_id']
    secondary = pd.read_csv(secondary_set_path)['track_id']

    d_s = {s: secondary_region for s in secondary}
    d_p = {p: primary_region for p in primary}

    d_s.update(d_p)

    result_df = pd.DataFrame({'track_id': d_s.keys(), 'region': d_s.values()})
    print(result_df)
    result_df.to_csv(output_path)
    return

def get_metrices_from_track_history():
    pass

if __name__ == '__main__':
    # pylint: disable=line-too-long
    CHART_PATH = 'D:\\Documents\\PROgram\\song-prediction\\data-collector\\data\\chart\\jp_weekly_totals.csv'
    STREAM_ROOT = 'D:\\Documents\\PROgram\\song-prediction\\data-collector\\data\\track\\streams\\'
