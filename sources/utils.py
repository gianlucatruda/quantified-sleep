"""Utility code for ingestion and processing"""

import pandas as pd


def pythonify(s):
    """Converts string into python format
    """
    return s.replace(' ', '_').replace(',', '').lower()


def restyle_columns(df: pd.DataFrame, prefix = '', style = 'pythonic', trunc = 30):

    if style != 'pythonic':
        raise NotImplementedError(style)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(type(df))

    _df=df.copy()

    _df.rename(
        {c: pythonify(f'{prefix}{c}')[:trunc] for c in _df.columns},
        axis=1, inplace=True)

    return _df
