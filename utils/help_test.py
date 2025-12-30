import pandas as pd

def filter_two_df_by_ms(df:pd.DataFrame,df2:pd.DataFrame):
    df = df[df['ms'].isin(df2['ms'])].copy()
    df2 = df2[df2['ms'].isin(df['ms'])].copy()
    df = df.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df['x'] = df.index
    df2['x'] = df2.index
    return df,df2