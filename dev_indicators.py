import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.draw_funcs import draw_hb_chart_fast
from indicators.classic_ind import *
from indicators.pva_ind import *
from indicators.arbitration_ind import *

raw_file = 'data_for_tests\data_from_moex5\_5IMOEXF_1_1763448583.parquet'
raw_file2 = 'data_for_tests\data_from_moex5\_5MMZ5_1_1763448587.parquet'
# raw_file = 'data_for_tests\data_from_moex5\_5IMOEXF_1_1763893692.parquet'
df = pd.read_parquet(raw_file)
df2 = pd.read_parquet(raw_file2)
# print(df.head())
# print(df2.head())
# print(len(df),len(df2))
# if len(df) != len(df2):
#     print(f"Предупреждение: разная длина df={len(df)}, df2={len(df2)}")
#     # Обрезаем до минимальной длины
#     min_len = min(len(df), len(df2))
#     df = df.iloc[:min_len]
#     df2 = df2.iloc[:min_len]
#     df = df.reset_index(drop=True)
#     df2 = df2.reset_index(drop=True)
def filter_two_df(df:pd.DataFrame,df2:pd.DataFrame):
    df = df[df['ms'].isin(df2['ms'])].copy()
    df2 = df2[df2['ms'].isin(df['ms'])].copy()
    df = df.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    return df,df2
print(len(df),len(df2))

disc = get_percent_diff_window(df,df2,147,10,'middle')
print(disc)
# df['middle150'] = df['middle'].shift(150)
# df2['middle150'] = df2['middle'].shift(150)

# df['diff'] = round(((df['middle'] - df['middle150'])/df['middle'])*100,2)
# df2['diff'] = round(((df2['middle'] - df2['middle150'])/df2['middle'])*100,2)
# df['disc'] = df['diff'] - df2['diff']
# print(df.head())
print(df.tail())
print(df2.tail())
if 1:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # sharex=True для синхронизации по оси X

    # Первый график
    plt.sca(ax1)
    plt.plot(df['middle'])
    plt.plot(df2['middle'])
    # draw_hb_chart_fast(df)

    # Второй график
    plt.sca(ax2)

    ax2.plot(disc)
else:
    ...
    # draw_hb_chart_fast(df)
    # plt.plot(df['bbu'],color='b')
    # plt.plot(df['bbd'],color='b')
    # plt.plot(df['bbm'],color='b')

    # plt.plot(df['diff'])
    # plt.plot(df2['diff'])
    plt.plot(disc)


# Автоматическая регулировка layout'а
plt.tight_layout()
plt.show()