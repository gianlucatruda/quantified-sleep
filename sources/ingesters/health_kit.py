"""Ingestion module for HealthKit daily/hourly csv dumps"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'hk_'


def ingest(fpath: str, sep=',', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep)

    # Convert types as appropriate
    df['Start'] = pd.to_datetime(df['Start'])
    df['Finish'] = pd.to_datetime(df['Finish'])

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    if debug_mode:
        logger.debug(df.info())

    return df
