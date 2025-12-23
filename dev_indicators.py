import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.draw_funcs import draw_hb_chart_fast
from indicators.classic_ind import *
from indicators.pva_ind import *

raw_file = 'data_for_tests\data_from_moex5\_5IMOEXF_1_1763448583.parquet'
# raw_file = 'data_for_tests\data_from_moex5\_5IMOEXF_1_1763893692.parquet'
df = pd.read_parquet(raw_file)

# df = add_bollinger(df,20)
# df['bbm'] = df['sma'].copy()
# df = add_sma(df,5)
# df['buff_bb'] = (df['bbu'] - df['bbd'])/2
# df['top_line'] = df['sma'] + df['buff_bb']
# df['bottom_line'] = df['sma'] - df['buff_bb']
df = add_atr(df,5)
df['prev_close'] = df['close'].shift(1)
df['top_line'] = df['prev_close'] + df['atr'] * 1
df['bottom_line'] = df['prev_close'] - df['atr'] * 1
        
print(df.tail())
if 0:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # sharex=True для синхронизации по оси X

    # Первый график
    plt.sca(ax1)
    draw_hb_chart_fast(df)

    # Второй график
    plt.sca(ax2)

    ax2.plot(df['atr'])
else:
    draw_hb_chart_fast(df)
    # plt.plot(df['bbu'],color='b')
    # plt.plot(df['bbd'],color='b')
    # plt.plot(df['bbm'],color='b')

    # plt.plot(df['sma'],color='g')
    plt.plot(df['top_line'],color='g')
    plt.plot(df['bottom_line'],color='g')


# Автоматическая регулировка layout'а
plt.tight_layout()
plt.show()