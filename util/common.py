import os
import re
from typing import  Union
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATE_PATTERN = r'(\d{4})[\/\\-](\d{2})[\/\\-](\d{2})'
def parse_date(date_string: str) -> Union[date, None]:
    match = re.match(DATE_PATTERN, date_string)
    
    if match:
        return date(int(match[1]), int(match[2]), int(match[3]))
    return None

def plot_track(track_id: str, stream_root: str):
    track_streams = load_track(track_id, 'jp', stream_root)
    # print(track_df)
    plt.plot(list(range(len(track_streams))), track_streams)
    # plt.xlim((0, 100))
    plt.show()
    return

class InvalidDateException(BaseException):
    pass

def load_track(track_id: str, use_region: str, stream_root: str):
    track_df = pd.read_csv(
        os.path.join(stream_root, f'{track_id}.csv'),
        usecols=['date', use_region.lower()]
    )
    track_df['date'] = track_df['date'].map(parse_date)

    start_date: date = track_df['date'][0]
    # assert (start_date - release_date).days >= 0,\
    #     f'track_id: {track_id}, start_date: {start_date}, release_date: {release_date}'
    
    end_date: date = track_df['date'][len(track_df) - 1]

    num_weeks = (end_date - start_date) / timedelta(days=7)

    # assert num_weeks.is_integer()
    if not num_weeks.is_integer():
        raise InvalidDateException

    num_weeks = int(num_weeks) + 1

    streams = np.empty([num_weeks])
    streams_idx = 0

    for cur_date, stream in zip(track_df['date'], track_df[use_region]):

        stream: float

        while cur_date > streams_idx * timedelta(days=7) + start_date:
            streams[streams_idx] = 0.0
            streams_idx += 1

        # assert cur_date == streams_idx * timedelta(days=7) + start_date
        if cur_date != streams_idx * timedelta(days=7) + start_date:
            raise InvalidDateException

        streams[streams_idx] = 0.0 if np.isnan(stream) else math.log10(stream)
        streams_idx += 1

    return streams
