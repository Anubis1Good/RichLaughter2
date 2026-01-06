import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.help_test import filter_two_df_by_ms
from utils.draw_funcs import draw_hb_chart_fast
from indicators.classic_ind import *
from indicators.pva_ind import *
from indicators.arbitration_ind import *

raw_file = 'data_for_tests\data_from_moex5\_5IMOEXF_1_1763893692.parquet'
raw_file2 = 'data_for_tests\data_from_moex5\_5MMZ5_1_1763893699.parquet'
# raw_file = 'data_for_tests\data_from_moex5\_5GLDRUBF_1_1767639808.parquet'
# raw_file2 = 'data_for_tests\data_from_moex5\_5GLH6_1_1767639810.parquet'

df = pd.read_parquet(raw_file)
df2 = pd.read_parquet(raw_file2)


print(len(df),len(df2))
df,df2 = filter_two_df_by_ms(df,df2)
# disc = get_percent_diff_window(df,df2,145,None,'close')
# print(disc)
df = add_bollinger(df)

print(df.tail())
print(df2.tail())
if 0:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # sharex=True для синхронизации по оси X

    # Первый график
    plt.sca(ax1)
    plt.plot(df['middle'])
    plt.plot(df2['middle'])
    # draw_hb_chart_fast(df)

    # Второй график
    plt.sca(ax2)

    # ax2.plot(disc)
else:
    ...
    draw_hb_chart_fast(df)
    plt.plot(df['bbu'],color='b')
    plt.plot(df['bbd'],color='b')
    plt.plot(df['sma'],color='b')
    # plt.plot(df['pdf_up'],color='g')
    # plt.plot(df['pdf_down'],color='r')




# Автоматическая регулировка layout'а
plt.tight_layout()
plt.show()