import pandas as pd
import numpy as np

def add_vodka_channel(df:pd.DataFrame,period=20):
    '''add top_mean, bottom_mean, avarege_mean'''
    df['top_mean'] = df['high'].rolling(window=period).median()
    df['bottom_mean'] = df['low'].rolling(window=period).median()
    df['avarege_mean'] = (df['top_mean'] + df['bottom_mean']) / 2
    return df

def add_average_fractals(df:pd.DataFrame, period=5):
    """add 'ave_up', 'ave_down'"""
    up_points = df[df['fractal_up']]
    df['ave_up'] = up_points['high'].rolling(window=period).mean()
    df['ave_up'] = df['ave_up'].ffill()
    down_points = df[df['fractal_down']]
    df['ave_down'] = down_points['low'].rolling(window=period).mean()
    df['ave_down'] = df['ave_down'].ffill()
    return df

def add_extremes_fractals(df:pd.DataFrame, period=5):
    """add 'ext_up', 'ext_down'"""
    up_points = df[df['fractal_up']]
    df['ext_up'] = up_points['high'].rolling(window=period).max()
    df['ext_up'] = df['ext_up'].ffill()
    down_points = df[df['fractal_down']]
    df['ext_down'] = down_points['low'].rolling(window=period).min()
    df['ext_down'] = df['ext_down'].ffill()
    return df

#TAKE THIS
def add_kvas_channel(df:pd.DataFrame,period=20):
    """add 'top_kvas','low_kvas'"""
    df['delta_p'] = (df['close'] - df['close'].shift(period))
    df['top_kvas'] = df['high'].rolling(period).max() + df['delta_p']
    df['low_kvas'] = df['low'].rolling(period).min() + df['delta_p']
    return df

#NEED EXPERIMENT
def add_kefir_channel(df:pd.DataFrame,period=20):
    """add 'top_kefir','low_kefir'"""
    df['delta_p'] = df['close'] - df['close'].shift(period)
    df['delta_t'] = df['delta_p'].shift(period).rolling(period).max()
    df['delta_l'] = df['delta_p'].shift(period).rolling(period).min()
    df['top_kefir'] = df['high'].rolling(period).max() + df['delta_t']
    df['low_kefir'] = df['low'].rolling(period).min() + df['delta_l']
    return df

def add_static_channel(df:pd.DataFrame,period=60):
    """add 'center_line', 'top_line', 'bottom_line'"""
    df['center_line'] = df['close'].rolling(period,1).quantile(0.5)
    df['top_line'] = df['close'].rolling(period,1).quantile(0.9)
    df['bottom_line'] = df['close'].rolling(period,1).quantile(0.1)
    return df

def add_nlevels_fractal(df,n=3,min_step=0.1,buff=0.05,offset=1):
    """add top_n and bot_n"""
    df = df.copy()
    top_lvl_raw = []
    bot_lvl_raw = []
    top_lvl = []
    bot_lvl = []
    for i in range(len(df)-1,0,-1):
        row = df.iloc[i]
        if row['fractal_up']:
            if row['high'] >= df['high'].iloc[i:len(df)-offset-1].max():
                top_lvl_raw.append(row['high']-row['high']*buff/100)
        if row['fractal_down']:
            if row['low'] <= df['low'].iloc[i:len(df)-offset-1].min():
                bot_lvl_raw.append(row['low']+row['low']*buff/100)
    for lvl in top_lvl_raw:
        if len(top_lvl) == 0:
            top_lvl.append(lvl)
        elif len(top_lvl) >= n:
            break
        elif abs(1 - lvl/top_lvl[-1])*100 > min_step:
            top_lvl.append(lvl)
    for lvl in bot_lvl_raw:
        if len(bot_lvl) == 0:
            bot_lvl.append(lvl)
        elif len(bot_lvl) >= n:
            break
        elif abs(1 - bot_lvl[-1]/lvl)*100 > min_step:
            bot_lvl.append(lvl)
    for i,lvl in enumerate(top_lvl):
        df['top_'+str(i)] = lvl
    for i,lvl in enumerate(bot_lvl):
        df['bot_'+str(i)] = lvl
    return df