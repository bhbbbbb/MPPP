import os

BASE_URL = 'https://kworb.net/spotify/'

CHART_BASE_URL = os.path.join(BASE_URL, 'country')

# pylint: disable=line-too-long
COUNTRIES = ['Global', 'US', 'MX', 'DE', 'GB', 'BR', 'AU', 'FR', 'CA', 'NL', 'ES', 'IT', 'SE', 'NO', 'AR', 'ID', 'CL', 'TR', 'PH', 'PL', 'BE', 'IN', 'DK', 'CH', 'IE', 'NZ', 'PE', 'CO', 'FI', 'SG', 'MY', 'AT', 'CZ', 'PT', 'TW', 'JP', 'EC', 'HU', 'IL', 'CR', 'GT', 'SA', 'ZA', 'HK', 'TH', 'PY', 'VN', 'BO', 'UY', 'SK', 'RO', 'GR', 'SV', 'AE', 'PA', 'HN', 'LT', 'IS', 'EG', 'DO', 'LU', 'LV', 'MA', 'EE', 'NI', 'BG', 'MT', 'CY', 'RU', 'UA', 'KR', 'AD']

def weekly_chart_url(country: str):

    country = country.lower()
    assert country in [con.lower() for con in COUNTRIES]

    return f'{CHART_BASE_URL}/{country}_weekly_totals.html'

BASE_TRACK_URL = os.path.join(BASE_URL, 'track')
