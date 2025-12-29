import os
import matplotlib.pyplot as plt
from traders.TestTrader.TestTrader import TestTrader
# from wss.help_wss.OpenWS import OpenWSCondition as WSS
from wss.LWS.LWS2a import LWS8_GRAVITON as WSS
# from wss.APSWS.APSWS1a import APSWS1_ as WSS

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
# symbols = ('CNYRUBF','CRZ5')

charts = {s: None for s in symbols}
for chart in charts_list:
    for s in symbols:
        if s in chart:
            charts[s] = os.path.join(folder_charts,chart)
tt1 = TestTrader(
    symbols,
    ('5min',),
    (1,),
    (
        WSS,    
        {
            'start':2500,
            'end':2850,
            'amount_lvl': 5,
            'uh_lvl': 3100,
            'dh_lvl': 2400,
            'first_long': False,
            'keep_hedge':True,
            'keep_pos':False,
            'last_point':True,
            'keep_start_pos':True
        }
    ),
    charts={'5min':charts},

    close_on_time=False

)

tt1.reload_data()
tt1.trade_data['IMOEXF']['pos'] = -5
tt1.trade_data['IMOEXF']['mp'] = 2400
tt1.trade_data['MMZ5']['pos'] = 5
tt1.trade_data['MMZ5']['mp'] = 2450
tt1.check_window_fast()
# # Печать статистики
tt1.print_statistics(symbols[0])
tt1.print_statistics(symbols[1])
full_total = tt1.trade_data[symbols[0]]['step_eq_vtb'][-1] + tt1.trade_data[symbols[1]]['step_eq_vtb'][-1]
print("Общая прибыль:",full_total)
tt1.plot_chart_and_sequtity(symbols[0],help_info='couple_pos')
tt1.plot_chart_and_sequtity(symbols[0],help_info='couple_complex')
# df = tt1.charts[symbols[0]]


