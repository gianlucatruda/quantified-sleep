"""Ingestion module for HealthKit activities collected with AWARE"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'aw_activ_'


def ingest(fpath: str, sep=',', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep)

    # Drop redundant column
    df.drop('activities', axis=1, inplace=True)

    # Convert types as appropriate
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    # Sort columns alphabetically
    df.sort_index(axis=1, inplace=True)

    if debug_mode:
        logger.debug(df.info())

    return df
