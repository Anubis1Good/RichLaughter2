import pandas as pd


def get_percent_diff_window(df:pd.DataFrame,df2:pd.DataFrame,window:int,smooth:int|None=None,kind='close'):
    """"""
    if df['ms'].iloc[-1] != df2['ms'].iloc[-1]:
        # print(df['ms'].iloc[-1],df2['ms'].iloc[-1])
        return None
    df = df.copy()
    df2 = df2.copy()
    df['kind_n'] = df[kind].shift(window)
    df2['kind_n'] = df2[kind].shift(window)
    df['diff'] = round(((df[kind] - df['kind_n'])/df[kind])*100,2)
    df2['diff'] = round(((df2[kind] - df2['kind_n'])/df2[kind])*100,2)
    disc = df['diff']- df2['diff']
    if smooth:
        disc = disc.rolling(smooth).mean()
    disc = disc.ffill()
    disc = disc.fillna(0)
    return disc
