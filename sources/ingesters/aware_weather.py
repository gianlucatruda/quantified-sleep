"""Ingestion module for weather collected with AWARE"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'aw_weather_'


def ingest(fpath: str, sep=',', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep)

    # Convert types as appropriate
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['sunrise'] = pd.to_datetime(df['sunrise']).dt.time
    df['sunset'] = pd.to_datetime(df['sunset']).dt.time

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    # Sort columns alphabetically
    df.sort_index(axis=1, inplace=True)

    if debug_mode:
        logger.debug(df.info())

    return df
