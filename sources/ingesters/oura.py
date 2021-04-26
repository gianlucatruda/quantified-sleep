"""Ingestion module for Oura ring JSON dumps"""

from loguru import logger
import pandas as pd
import numpy as np
import json
import os
import sys
from ..utils import restyle_columns

PREFIX = 'oura_'


def ingest(fpath: str, debug_mode=False) -> pd.DataFrame:

    # Read in the JSON file
    with open(fpath, 'r') as read_file:
        data = json.load(read_file)

    # Created a dataframe for each section of the ingested data
    dframes = {k: pd.DataFrame(data[k]) for k in data.keys()}

    # Retrieve sleep and readiness data
    df_sleep = dframes['sleep'].copy()
    df_ready = dframes['readiness'].copy()

    # Convert types as appropriate
    df_sleep['summary_date'] = pd.to_datetime(df_sleep['summary_date'])
    df_ready['summary_date'] = pd.to_datetime(df_ready['summary_date'])

    for c in ['bedtime_start', 'bedtime_end']:
        df_sleep[f'{c}_utc'] = pd.to_datetime(df_sleep[c], utc=True)

    # Perform renaming and remove spaces
    df_sleep = restyle_columns(df_sleep, prefix=PREFIX)
    df_ready = restyle_columns(df_ready, prefix=PREFIX + 'ready_')

    # Sort columns alphabetically
    df_sleep.sort_index(axis=1, inplace=True)
    df_ready.sort_index(axis=1, inplace=True)

    if debug_mode:
        logger.debug(df_sleep.info())
        logger.debug(df_ready.info())

    return df_sleep, df_ready
