from wss.LWS.LWS2 import LWS5_XPG,LWS5_EUCLID,LWS5_MERCATUS
from wss.LWS.LWS2a import LWS8_SINGULARITY
from wss.APSWS.APSWS1a import APSWS1_DYNAMO

from traders.QuikTrader.QuikTrader import QuikTrader

bot_on_ticker = [
    {
        'ws': LWS8_SINGULARITY,
        'ws_params':{
            'start':10.980,
            'end':11.140,
            'amount_lvl': 4,
            'uh_lvl': 11.200,
            'dh_lvl': 10.900,
            'first_long': False,
            'keep_hedge':True,
            'keep_pos':False,
            'last_point':True
        },
        'dts': [
            {
                'ss':('CNYRUBF','CRH6',),
                'tfs':('M5',),
                'qs': (1,1,)
            }
        ]

    },
    {
        'ws': APSWS1_DYNAMO,
        'ws_params':{
            'first_long': True,
            'funding': True,
            'hour_fund':18,
            'minute_fund':20
        },
        'dts': [
            {
                'ss':('GLDRUBF','GLH6',),
                'tfs':('M5',),
                'qs': (1,1,)
            }
        ]

    },
    # {
    #     'ws': LWS5_MERCATUS,
    #     'ws_params':{
    #         'start':10.660,
    #         'end':11.300,
    #         'amount_lvl': 5,
    #         'us_lvl': 11.430,
    #         'ds_lvl': None,
    #         'grid_dir': 1,
    #         'hold_pos': False
    #     },
    #     'dts': [
    #         {
    #             'ss':('CNYRUBF',),
    #             'tfs':('M5',),
    #             'qs': (1,)
    #         }
    #     ]

    # },
    # {
    #     'ws': LWS5_EUCLID,
    #     'ws_params':{
    #         'lvls':(2515,2625,2720),
    #         'us_lvl': 2770,
    #         'ds_lvl': None,
    #         'grid_dir': 1,
    #         'hold_pos': False
    #     },
    #     'dts': [
    #         {
    #             'ss':('IMOEXF',),
    #             'tfs':('M5',),
    #             'qs': (1,)
    #         }
    #     ]

    # },
    # {
    #     'ws': LWS5_EUCLID,
    #     'ws_params':{
    #         'lvls':(118.78,124.02),
    #         'us_lvl': 130.51,
    #         'ds_lvl': None,
    #         'grid_dir': 1,
    #         'hold_pos': False
    #     },
    #     'dts': [
    #         {
    #             'ss':('GAZPF',),
    #             'tfs':('M5',),
    #             'qs': (1,)
    #         }
    #     ]
    # },
    
    # {
    #     'ws': LWS2_PSGSON,
    #     'ws_params':{
    #         'amount_lvl': 3,
    #         'per_step':0.10,
    #         'keep':False,
    #         'reset_n':3
    #     },
    #     'dts': [
    #         {
    #             'ss':('IMOEXF','MMZ5'),
    #             'tfs':('M5',),
    #             'qs': (1,1)
    #         }
    #     ]

    # },
    # {
    #     'ws': LWS2_SWIMIGSON,
    #     'ws_params':{
    #         'amount_lvl': 3,
    #         'per_step':0.5,
    #         'grid_dir': 1,
    #         'keep':False,
    #         'reset_n':2
    #     },
    #     'dts': [
    #         {
    #             'ss':('GZZ5',),
    #             'tfs':('M5',),
    #             'qs': (1,)
    #         }
    #     ]

    # },
]

def init_trader() -> list[QuikTrader]:
    bots = []
    for conf_ws in bot_on_ticker:
        ws = (conf_ws['ws'],conf_ws['ws_params'])
        for dt in conf_ws['dts']:
            close_on_time = dt.get('close_on_time',False)
            cur_margin = dt.get('cur_margin',True)
            stop_risk = dt.get('stop_risk',None)
            print(dt['ss'],dt['tfs'],ws,'close_on_time:',close_on_time,'stop_risk:',stop_risk,'cur_margin',cur_margin)
            bot = QuikTrader(dt['ss'],dt['tfs'],dt['qs'],ws,need_debug=True,close_on_time=close_on_time,stop_risk=stop_risk,cur_margin=cur_margin)
            bots.append(bot)
    return bots