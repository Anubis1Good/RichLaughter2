from wss.LWS.LWS2 import LWS5_XPG,LWS5_EUCLID,LWS5_MERCATUS


from traders.QuikTrader.QuikTrader import QuikTrader

bot_on_ticker = [
    # {
    #     'ws': LWS1_AUTOGRID,
    #     'ws_params':{
    #         'start':11.440,
    #         'end':11.640,
    #         'amount_lvl': 10,
    #         'us_lvl': 11.740,
    #         'ds_lvl': 11.340,
    #         'grid_dir': 0,
    #     },
    #     'dts': [
    #         {
    #             'ss':('CRZ5',),
    #             'tfs':('M5',),
    #             'qs': (1,)
    #         }
    #     ]

    # },
    {
        'ws': LWS5_MERCATUS,
        'ws_params':{
            'start':10.660,
            'end':11.270,
            'amount_lvl': 5,
            'us_lvl': 11.450,
            'ds_lvl': None,
            'grid_dir': 1,
            'hold_pos': False
        },
        'dts': [
            {
                'ss':('CNYRUBF',),
                'tfs':('M5',),
                'qs': (1,)
            }
        ]

    },
    {
        'ws': LWS5_EUCLID,
        'ws_params':{
            'lvls':(2515,2625,2720),
            'us_lvl': 2770,
            'ds_lvl': None,
            'grid_dir': 1,
            'hold_pos': False
        },
        'dts': [
            {
                'ss':('IMOEXF',),
                'tfs':('M5',),
                'qs': (1,)
            }
        ]

    },
    # {
    #     'ws': LWS5_CADUCEUS,
    #     'ws_params':{
    #         'start':2608,
    #         'end':2675,
    #         'amount_lvl': 3,
    #         'us_lvl': 2690,
    #         'ds_lvl': None,
    #         'grid_dir': 1,
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
            print(dt['ss'],dt['tfs'],ws)
            bot = QuikTrader(dt['ss'],dt['tfs'],dt['qs'],ws,need_debug=True,close_on_time=False)
            bots.append(bot)
    return bots