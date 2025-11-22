import pandas as pd
import matplotlib.pyplot as plt
from utils.draw_funcs import draw_hb_chart_fast
from indicators.classic_ind import add_atr

raw_file = 'data_for_tests\data_from_moex5\_5IMOEXF_1_1763806893.parquet'
df = pd.read_parquet(raw_file)


df = add_atr(df,5)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # sharex=True для синхронизации по оси X

# Первый график
plt.sca(ax1)
draw_hb_chart_fast(df)

# Второй график
plt.sca(ax2)

ax2.plot(df['atr'])


# Автоматическая регулировка layout'а
plt.tight_layout()
plt.show()