"""Ingestion module for Way of Life csv dumps"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'wol_'


def ingest(fpath: str, sep=',', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep)

    # Convert types as appropriate
    df['Date'] = pd.to_datetime(df['Date'])

    def string_to_bool(s):
        """ Converts 'Yes' and 'No' to boolean ints
        """
        s = s.lower()
        if s in ['yes', 'true', 'skip']:
            return 1.0
        elif s in ['no', 'false']:
            return 0.0
        else:
            return np.NaN

    bool_cols = list(df.columns)
    bool_cols.remove('Date')
    for c in bool_cols:
        df[c] = df[c].apply(string_to_bool)

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    # Sort columns alphabetically
    df.sort_index(axis=1, inplace=True)

    if debug_mode:
        logger.debug(df.info())

    return df
