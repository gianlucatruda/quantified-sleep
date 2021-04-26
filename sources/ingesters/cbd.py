"""Ingestion module for CBD oil csv dumps"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'cbd_'


def ingest(fpath: str, sep=',', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep)

    # Convert types as appropriate
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    if debug_mode:
        logger.debug(df.info())

    return df
