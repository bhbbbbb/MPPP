import re
from typing import List

import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd

from .url import BASE_TRACK_URL

class TrackParser:

    @staticmethod
    def fetch(track_id: str):

        # url = os.path.join(BASE_TRACK_URL, f'{track_id}.html')
        url = BASE_TRACK_URL + f'/{track_id}.html'

        with requests.get(url) as res:

            assert res.status_code == 200, f'got status: {res.status_code}, when getting: {url}'

            res.encoding = res.apparent_encoding
            text = res.text

            # with open('jp_weekly_totals.html', 'w', encoding='utf8') as fout:
            #     fout.write(text)
            return text

    @staticmethod
    def parse(html: str):
        """

        Args:
            html (str): 

        Returns:
            pos_df, stream_df
        """
        
        soup = BeautifulSoup(html, 'html.parser')

        rows = soup.table.find_all('tr')

        headings = [ heading.string.lower() for heading in rows[0].find_all('th') ][1:]
        # totals = [ remove_comma(total.string) for total in rows[1].find_all('td')[1:] ]

        # peak_row: List[Tag] = rows[2].find_all('td')[1:]

        # peak_pos = [ int(peak.find_next('span', class_='p').string) for peak in peak_row ]
        # peak_streams = [ remove_comma(peak.find_next('span', class_='s').string) for peak in peak_row]

        tem = [TrackParser._parse_row(row, headings) for row in rows[3:]]

        pos_rows, stream_row = list(zip(*tem))

        # print(pos_rows)
        # df = pd.DataFrame(rows, columns=[con.lower() for con in COUNTRIES])
        pos_df = pd.DataFrame(pos_rows)
        stream_df = pd.DataFrame(stream_row)

        return pos_df, stream_df


    @staticmethod
    def _remove_comma(long_int: str):
        return int(re.sub(',', '', long_int))

    @staticmethod
    def _parse_row(tr_tag: Tag, headings: List[str]):
        cols: List[Tag] = tr_tag.find_all('td')

        date = cols[0].string
        pos_row = dict(
            date = date,
        )

        stream_row = dict(
            date = date,
        )

        for col, heading in zip(cols[1:], headings):
            if col.string == '--':
                pos_row[heading] = stream_row[heading] = None
            else:
                pos_row[heading] = int(col.find_next('span', class_='p').string)
                stream_row[heading] = TrackParser._remove_comma(col.find_next('span', class_='s').string)

        return pos_row, stream_row
