"""Ingestion module for daily metrics spreadsheet"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'daily_'


def ingest(fpath: str, sep='\t', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep)

    # Convert types as appropriate
    df['Date'] = pd.to_datetime(df['Date'])

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    # Sort columns alphabetically
    df.sort_index(axis=1, inplace=True)

    if debug_mode:
        logger.debug(df.info())

    return df
