"""Ingestion module for nomie 3 csv dumps"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'nomie_'


def ingest(fpath: str, sep=',', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep)

    # Extract important columns
    df = df[['start', 'end', 'tracker', 'value']]

    # Convert types as appropriate
    for c in ['start', 'end']:
        df[c] = pd.to_datetime(df[c])

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    # Sort columns alphabetically
    df.sort_index(axis=1, inplace=True)

    if debug_mode:
        logger.debug(df.info())

    return df
