import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.help_test import filter_two_df_by_ms
from traders.TestTrader.TestTrader import TestTrader
# from wss.help_wss.OpenWS import OpenWSCondition as WSS
from wss.LWS.LWS2a import LWS8_LITE as WSS
# from wss.APSWS.APSWS1a import APSWS2_SPARTACUS as WSS

#     {
#     'period':50,
#     'amount_lvl': 2,
#     'grid_dir': 0,
#     'per_limit': 0.1,
#     'keep': True
# }
            # 'amount_lvl': 4,
            # 'per_step':0.05,
            # 'grid_dir': 0,
            # 'keep':False,
            # 'reset_n': 3

folder_charts = 'data_for_tests\data_from_moex5'
charts_list = os.listdir(folder_charts)
symbols = ('IMOEXF','MMZ5')
quantities = [1 for s in symbols]
# symbols = ('CNYRUBF','CRZ5')
# symbols = ('GLDRUBF','GLH6')

charts = {s: None for s in symbols}
for chart in charts_list:
    for s in symbols:
        if s in chart:
            charts[s] = os.path.join(folder_charts,chart)
tt1 = TestTrader(
    symbols,
    ('5min',),
    quantities,
    (
        WSS,    
        {
            'start_lvl':2550,
            'end_lvl':2700,
            'uh_lvl': 2860,
            'dh_lvl': 2500,
            'first_long': False,
            'keep_hedge':True,
            'last_point':True,
            'keep_start_long':False,
            'keep_start_short':False,
            'close_hedge':True
        }
    ),
    charts={'5min':charts},

    close_on_time=False

)



df = tt1.charts[tt1.timeframes[0]][symbols[0]]
df2 = tt1.charts[tt1.timeframes[0]][symbols[1]]
print(len(df),len(df2))
tt1.charts[tt1.timeframes[0]][symbols[0]], tt1.charts[tt1.timeframes[0]][symbols[1]] = filter_two_df_by_ms(df,df2)
print(len(tt1.charts[tt1.timeframes[0]][symbols[0]]),len(tt1.charts[tt1.timeframes[0]][symbols[1]]))
tt1.reload_data()
# tt1.trade_data['IMOEXF']['pos'] = -5
# tt1.trade_data['IMOEXF']['mp'] = 2400
# tt1.trade_data['MMZ5']['pos'] = 5
# tt1.trade_data['MMZ5']['mp'] = 2450
tt1.check_window_fast()
# # Печать статистики
tt1.print_statistics(symbols[0])
tt1.print_statistics(symbols[1])
full_total = tt1.trade_data[symbols[0]]['step_eq_vtb'][-1] + tt1.trade_data[symbols[1]]['step_eq_vtb'][-1]
print("Общая прибыль:",full_total)
tt1.plot_chart_and_sequtity(symbols[0],help_info='couple_pos')
tt1.plot_chart_and_sequtity(symbols[0],help_info='couple_complex')
# df = tt1.charts[symbols[0]]


