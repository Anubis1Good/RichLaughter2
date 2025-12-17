import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.draw_funcs import draw_hb_chart_fast
from indicators.classic_ind import add_fractals
from indicators.pva_ind import add_nlevels_fractal

# raw_file = 'data_for_tests\data_from_moex5\_5IMOEXF_1_1763448583.parquet'
raw_file = 'data_for_tests\data_from_moex5\_5IMOEXF_1_1763893692.parquet'
df = pd.read_parquet(raw_file)


df = add_fractals(df,10)

    
df = add_nlevels_fractal(df,10,0.5)
        
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
    ups = df['fractal_up'] == True
    downs = df['fractal_down'] == True
    plt.scatter(df.loc[ups, 'x'], df.loc[ups, 'high'])
    plt.scatter(df.loc[downs, 'x'], df.loc[downs, 'low'])
    # print(df.iloc[-1].index)
    for col in df.columns:
        if 'top_' in col:
            plt.plot(df[col],color='r')
        if 'bot_' in col:
            plt.plot(df[col],color='g')
    # plt.plot(df['fractal_down'])


# Автоматическая регулировка layout'а
plt.tight_layout()
plt.show()