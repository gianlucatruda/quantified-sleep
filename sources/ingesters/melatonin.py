"""Ingestion module for melatonin csv dumps"""

from loguru import logger
import pandas as pd
import numpy as np
import os
import sys
from ..utils import restyle_columns

PREFIX = 'melatonin_'


def ingest(fpath: str, sep=',', debug_mode=False) -> pd.DataFrame:

    # Read in the file as csv/tsvd
    df = pd.read_csv(fpath, sep=sep, header=None)

    # Extract important columns
    df.columns = ['Date']

    # Convert types as appropriate
    df['Date'] = pd.to_datetime(df['Date'])

    df['Quantity'] = df['Date'].apply(lambda x: 1)

    # Perform renaming and column name restructuring
    df = restyle_columns(df, prefix=PREFIX)

    if debug_mode:
        logger.debug(df.info())

    return df
