from datetime import date
import os
import math
from typing import Callable, List, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util.track import Track


def plot_tracks(
    tracks: List[Track],
    title: str,
    transform: Callable[[np.ndarray], Union[float, int]],
    sort: bool = True,
    output_dir: str = None,
):
    """_summary_

    Args:
        output_dir (str, optional): Ignore to show figs without saving.
    """

    streams = [transform(track.stream) for track in tracks]

    if sort:
        streams.sort(reverse=True)

    plt.figure(figsize=(16, 9))
    plt.title(title)
    plt.plot(list(range(len(tracks))), streams)
    if sort:
        plt.annotate(f'max-{title}={streams[0]}', (0, streams[0]))
        plt.annotate(f'min-{title}={streams[-1]}', (len(tracks) - 1, streams[-1]))
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{title}.png'))
    else:
        plt.show()
    return

def plot_stats(
    set_path: str,
    use_region: str,
    stream_root: str,
    now_date: date,
    output_dir: str = None
):

    df = pd.read_csv(set_path)

    track_ids = df['track_id']
    tracks = [
        Track(track_id, use_region=use_region, stream_root=stream_root, now_date=now_date) \
            for track_id in track_ids
    ]
    
    def plot_(title: str, transform: Callable[[np.ndarray], Union[float, int]]):
        plot_tracks(tracks, f'{title}-sorted', transform, True, output_dir)
        plot_tracks(tracks, title, transform, False, output_dir)

    plot_('length', len)
    plot_('maximum', lambda x: x.max())
    plot_('minimum', lambda x: x.min())
    plot_('sumation', lambda x: math.log10(np.vectorize(lambda a: 10 ** a)(x).sum()))
    plot_('debut', lambda x: x[0])
    return
