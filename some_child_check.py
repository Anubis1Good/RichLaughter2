import os
from traders.TestTrader.TestTrader import TestTrader
from wss.LWS.LWS1 import LWS2_SWIMIGSON as WSS
# from wss.LWS.LWS1 import LWS2_SWIMGRID as WSS
# from wss.LWS.LWS1 import LWS2_PSG as WSS
# from wss.LWS.LWS1 import LWS2_PSGSON as WSS
# from wss.LWS.LWS1 import LWS3_APEX as WSS
# from wss.PWS.PWS1 import PWS1_GRIDC as WSS

folder_charts = 'data_for_tests\data_from_moex'
charts_list = os.listdir(folder_charts)
symbols = ('IMOEXF','MMZ5')
charts = {s: None for s in symbols}
for chart in charts_list:
    for s in symbols:
        if s in chart:
            charts[s] = os.path.join(folder_charts,chart)

tt1 = TestTrader(
    symbols,
    ('5min',),
    (1,1),
    (
        WSS,    
        {
            'amount_lvl': 4,
            'per_step':0.5,
            'grid_dir': 0,
            'keep':False,
            'reset_n':2
    }
    ),
    charts={'5min':charts},
    close_on_time=True

)

# tt1.check_child_lite()
tt1.check_child()
# # # Печать статистики
tt1.print_statistics('IMOEXF')
tt1.print_statistics('MMZ5')
# tt1.plot_equity('IMOEXF')
# tt1.plot_equity('MMZ5')
tt1.plot_chart('MMZ5',convert_tf='5min')



