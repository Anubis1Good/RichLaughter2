import os
from wss.LWS.LWS1 import *
from wss.PWS.PWS1 import *

tf_folders = {
    '1min': 'data_for_tests\data_from_moex',
    '5min': 'data_for_tests\data_from_moex5'
}

def get_df_file(symbol,tf):
    folder = tf_folders[tf]
    files = os.listdir(folder)
    for file in files:
        if symbol in file:
            return os.path.join(folder,file)
    raise FileNotFoundError(f"File for {symbol} with timeframe {tf} not found in {folder}")

def create_charts_confs(symbols,tfs):
    charts = {}
    for tf in tfs:
        charts[tf] = {}
        for s in symbols:
            charts[tf][s] = get_df_file(s,tf)
    return charts

group = {
    LWS2_SWIMIGSON: {
            'amount_lvl': [2, 3, 4, 5, 6],
            'per_step': [0.05, 0.1, 0.15, 0.2,10],
            'grid_dir': [-1, 0, 1],
            'keep': [True, False],
            'reset_n': [1, 2, 3,10]
       
    },

}

optimization_configs = [
    {
        'symbols': ['IMOEXF'],
        'timeframes': ['5min'],
        'ws_class': LWS2_SWIMIGSON,
        'ws_params_options': group[LWS2_SWIMIGSON],

    },

]
fee = 0.0002
close_on_time = True
for op in optimization_configs:
    op['charts'] = create_charts_confs(op['symbols'],op['timeframes'])
    op['quantity'] = [1] * len(op['symbols'])
    op['fee'] = fee
    op['close_on_time'] = close_on_time