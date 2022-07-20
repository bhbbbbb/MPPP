import os
from typing import Callable, List, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util.common import load_track


def plot(
    tracks: List[np.ndarray],
    title: str,
    transform: Callable[[np.ndarray], Union[float, int]],
    sort: bool = True,
):

    values = [transform(track) for track in tracks]

    if sort:
        values.sort(reverse=True)

    plt.figure(figsize=(16, 9))
    plt.title(title)
    plt.plot(list(range(len(tracks))), values)
    if sort:
        plt.annotate(f'max-{title}={values[0]}', (0, values[0]))
        plt.annotate(f'min-{title}={values[-1]}', (len(tracks) - 1, values[-1]))
    os.makedirs('data/stats', exist_ok=True)
    plt.savefig(f'data/stats/{title}.png')
    # plt.show()
    return

def plot_stats(set_path: str, stream_root: str):

    df = pd.read_csv(set_path)

    track_ids = df['track_id']
    tracks = [load_track(track_id, 'jp', stream_root) for track_id in track_ids]
    plot(tracks, 'length-sorted', len, True)
    plot(tracks, 'length', len, False)
    plot(tracks, 'maximum-sorted', lambda x: x.max(), True)
    plot(tracks, 'maximum', lambda x: x.max(), False)
    plot(tracks, 'minimum-sorted', lambda x: x.min(), True)
    plot(tracks, 'minimum', lambda x: x.min(), False)
    return



if __name__ == '__main__':
    SET_PATH = 'D:\\Documents\\PROgram\\song-prediction\\data-collector\\data\\sets\\all.csv'
    STREAM_ROOT = 'D:\\Documents\\PROgram\\song-prediction\\data-collector\\data\\track\\streams'
    plot_stats(SET_PATH, STREAM_ROOT)
