import re

import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd

from .url import weekly_chart_url

ID_PATTERN = r'.*[\/\\](\w+)\.html'

class ChartParser:
    
    @staticmethod
    def fetch(country: str):

        url = weekly_chart_url(country)

        with requests.get(url) as res:

            assert res.status_code == 200, f'got status: {res.status_code}, when getting: {url}'

            res.encoding = res.apparent_encoding
            text = res.text

            # with open('jp_weekly_totals.html', 'w', encoding='utf8') as fout:
            #     fout.write(text)
            return text

    @staticmethod
    def parse(html: str):

        # html = fetch(country)
        
        soup = BeautifulSoup(html, 'html.parser', from_encoding='utf8')

        children = soup.tbody.find_all('tr')
        rows = [ChartParser._parse_row(child) for child in children]

        df = pd.DataFrame(rows)
        return df

            

    @staticmethod
    def _remove_comma(long_int: str):
        return int(re.sub(',', '', long_int))

    @staticmethod
    def _parse_row(tr_tag: Tag):
        cols = tr_tag.find_all('td')

        artist, track = cols[1].div.find_all('a')

        return dict(
            pos = int(cols[0].string),
            artist_name = artist.string,
            artist_id = re.match(ID_PATTERN, artist['href'])[1],
            track_name = track.string,
            track_id = re.match(ID_PATTERN, track['href'])[1],
            weeks = ChartParser._remove_comma(cols[2].string),
            top10 = ChartParser._remove_comma(cols[3].string) if cols[3].string else -1,
            peak = ChartParser._remove_comma(cols[4].string),
            peak_x = ChartParser._remove_comma(
                    re.match(r'\(x(\d+)\)', cols[5].string)[1]
                ) if cols[5].string else -1,
            peak_streams = ChartParser._remove_comma(cols[6].string),
            totals = ChartParser._remove_comma(cols[7].string),
        )
