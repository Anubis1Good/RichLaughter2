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
    LWS2_SWIMGRID: {
            'amount_lvl': [2, 3, 4, 5],
            'per_step':[0.05, 0.06, 0.15, 0.2,10],
            'grid_dir': [-1, 0, 1],
            'keep':[True, False],
            'reset_n': [1, 2, 3,10],
    },
    LWS2_SWIMIGSON: {
            'amount_lvl': [2, 3, 4, 5],
            'per_step': [0.05, 0.1, 0.15, 0.2,10],
            'grid_dir': [-1, 0, 1],
            'keep': [True, False],
            'reset_n': [1, 2, 3,10],
       
    },
    LWS2_PSG: {
            'amount_lvl': [2, 3, 4, 5],
            'per_step':[0.05, 0.1, 0.15, 0.2,10],
            'keep':[True, False],
            'reset_n': [1, 2, 3,10],
    },
    LWS2_PSGSON: {
            'amount_lvl': [2, 3, 4, 5],
            'per_step':[0.05, 0.1, 0.15, 0.2,10],
            'keep':[True, False],
            'reset_n': [1, 2, 3,10],
    },
    LWS3_NEXUS: {
            'period_rsi':[5,7,14,140],
            'lvls':[(10,30,70,90),(5,25,75,95),(15,35,65,85)],
            'buff':[5,10],
    },
    LWS3_APEX: {
            'period_rsi':[5,7,14,140],
            'lvls':[(10,30,70,90),(5,25,75,95),(15,35,65,85)],
            'buff':[5,10],
    },
    LWS4_SWATR: {
            'amount_lvl': [2, 3, 4, 5],
            'atr_multiplier': [0.5,1.0,5.8,10.1],
            'atr_period': [5,7,14,140],
            'grid_dir': [-1, 0, 1],
            'keep': [True, False],
            'reset_n':  [1, 2, 3,10],
            'smoothing_factor': [0.01,0.1,1],
            'buffer_multiplier': [0.05,0.1,0.5],
    },
    PWS1_GRIDC: {
            'period':[5,7,14,140],
            'amount_lvl': [2, 3, 4, 5],
            'grid_dir': [-1, 0, 1],
            'per_limit':[0.05,0.1, 1],
            'keep': [True, False],
    },
    PWS1_PRGDC: {
            'period':[5,7,14,140],
            'amount_lvl': [2, 3, 4, 5],
            'per_limit': [0.05,0.1, 1],
            'keep': [True, False],
    }

}
symbols_tf_confs = [
    {
        'symbols': [
            ['IMOEXF'], ['MMZ5'], ['RMZ5'], ['SRZ5'], ['GAZPF'], 
            ['SBERF'], ['CRZ5'], ['CNYRUBF'], ['SVZ5'], ['GZZ5']
        ],
        'timeframes': [['5min']],
        'wss': (LWS2_SWIMGRID, LWS2_SWIMIGSON, LWS4_SWATR, PWS1_GRIDC)
    },
    {
        'symbols': [
            ['IMOEXF', 'MMZ5'], 
            ['SBERF', 'SRZ5'], 
            ['GAZPF', 'GZZ5'], 
            ['CNYRUBF', 'CRZ5']
        ],
        'timeframes': [['5min']],
        'wss': (LWS2_PSG, LWS2_PSGSON, LWS3_NEXUS, LWS3_APEX, PWS1_PRGDC)
    },
]

optimization_configs = []
for stc in symbols_tf_confs:
    for ws in stc['wss']:
        for s in stc['symbols']:
            for t in stc['timeframes']:
                conf = {
                    'symbols': s,
                    'timeframes': t,
                    'ws_class': ws,
                    'ws_params_options': group[ws],
                }
                optimization_configs.append(conf)

# Так было раньше
# optimization_configs = [
#     {
#         'symbols': ['IMOEXF'],
#         'timeframes': ['5min'],
#         'ws_class': LWS2_SWIMIGSON,
#         'ws_params_options': group[LWS2_SWIMIGSON],

#     },
#     {
#         'symbols': ['GZZ5'],
#         'timeframes': ['5min'],
#         'ws_class': PWS1_GRIDC,
#         'ws_params_options': group[PWS1_GRIDC],

#     },

# ]
fee = 0.0002
close_on_time = True
for op in optimization_configs:
    op['charts'] = create_charts_confs(op['symbols'],op['timeframes'])
    op['quantity'] = [1] * len(op['symbols'])
    op['fee'] = fee
    op['close_on_time'] = close_on_time

print(len(optimization_configs))