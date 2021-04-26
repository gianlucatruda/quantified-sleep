"""Ingestion module for RescueTime csv dumps"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'rescue_'


def ingest(fpath: str, sep=',', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep)

    # Convert types as appropriate
    df['Date'] = pd.to_datetime(df['Date'])

    time_cols = list(df.columns)
    time_cols.remove('Date')

    def convert_time(t: str):
        """Converts text-based times from RescueTime into datetimes
        """
        try:
            if t == 'no time':
                return pd.Timedelta(0)
            else:
                sep = ''
                format = ''
                if 's' in t:
                    format = '%Ss' + format
                    sep = ' '
                if 'm' in t:
                    format = f'%Mm{sep}' + format
                    sep = ' '
                if 'h' in t:
                    format = f'%Hh{sep}' + format
            return pd.Timedelta(t, format=format)
        except Exception as e:
            logger.warning(t, format)
            raise(e)

    for c in time_cols:
        df[c] = df[c].apply(convert_time)

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    # Sort columns alphabetically
    df.sort_index(axis=1, inplace=True)

    if debug_mode:
        logger.debug(df.info())

    return df
