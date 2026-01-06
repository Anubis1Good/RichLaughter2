import pandas as pd
import numpy as np

def add_big_volume(df:pd.DataFrame,period=20,multiplier=1):
    """add sma_volume, is_big """
    df['sma_volume'] = df['volume'].rolling(period).mean()
    df['is_big'] = df['volume']*multiplier > df['sma_volume']
    return df

def add_over_bb(df:pd.DataFrame):
    '''add over_bbu and over_bbd'''
    df['over_bbu'] = df['bbu'] < df['low']
    df['over_bbd'] = df['bbd'] > df['high']
    return df