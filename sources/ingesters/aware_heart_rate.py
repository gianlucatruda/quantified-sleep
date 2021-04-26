"""Ingestion module for heart rate collected (via HealthKit) with AWARE"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'aw_hr_'


def ingest(fpath: str, sep=',', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep)

    # Convert types as appropriate
    df['log_timestamp'] = pd.to_datetime(df['log_timestamp'])
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    # Group devices logically
    df['device'] = df['device'].apply(lambda x: 'Mi Smart Band 4' if 'Mi Smart Band 4' in x else x)

    # Drop useless column
    if 'type' in df.columns:
        df.drop('type', axis=1, inplace=True)

    # Organise by start time
    df = df.sort_values('start_time', ascending=True).reset_index(drop=True)

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    # Sort columns alphabetically
    df.sort_index(axis=1, inplace=True)

    if debug_mode:
        logger.debug(df.info())

    return df
