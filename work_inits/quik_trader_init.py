from wss.LWS.LWS1 import LWS1_FIRSTGRID,LWS1_AUTOGRID


from traders.QuikTrader.QuikTrader import QuikTrader

bot_on_ticker = [
    {
        'ws': LWS1_AUTOGRID,
        'ws_params':{
            'start':11.440,
            'end':11.640,
            'amount_lvl': 10,
            'us_lvl': 11.740,
            'ds_lvl': 11.340,
            'grid_dir': 0,
        },
        'dts': [
            {
                'ss':('CRZ5',),
                'tfs':('M5',),
                'qs': (1,)
            }
        ]

    },
    {
        'ws': LWS1_AUTOGRID,
        'ws_params':{
            'start':11.360,
            'end':11.440,
            'amount_lvl': 5,
            'us_lvl': 11.540,
            'ds_lvl': 11.260,
            'grid_dir': 0,
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
        'ws': LWS1_AUTOGRID,
        'ws_params':{
            'start':2537,
            'end':2550,
            'amount_lvl': 2,
            'us_lvl': 2557,
            'ds_lvl': None,
            'grid_dir': 1,
        },
        'dts': [
            {
                'ss':('MMZ5',),
                'tfs':('M5',),
                'qs': (1,)
            }
        ]

    },
    {
        'ws': LWS1_FIRSTGRID,
        'ws_params':{
            'lvls':(2490,2505,2513),
            'us_lvl': 2523,
            'ds_lvl': None,
            'grid_dir': 1 
        },
        'dts': [
            {
                'ss':('IMOEXF',),
                'tfs':('M5',),
                'qs': (1,)
            }
        ]

    },
    {
        'ws': LWS1_AUTOGRID,
        'ws_params':{
            'start':11885,
            'end':12080,
            'amount_lvl': 3,
            'us_lvl': 12120,
            'ds_lvl': None,
            'grid_dir': 1,
        },
        'dts': [
            {
                'ss':('GZZ5',),
                'tfs':('M5',),
                'qs': (1,)
            }
        ]

    },
]

def init_trader() -> list[QuikTrader]:
    bots = []
    for conf_ws in bot_on_ticker:
        ws = (conf_ws['ws'],conf_ws['ws_params'])
        for dt in conf_ws['dts']:
            print(dt['ss'],dt['tfs'],ws)
            bot = QuikTrader(dt['ss'],dt['tfs'],dt['qs'],ws,need_debug=True)
            bots.append(bot)
    return bots