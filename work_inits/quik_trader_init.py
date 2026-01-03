from wss.LWS.LWS2a import LWS8_SINGULARITY
from wss.APSWS.APSWS1a import APSWS1_DYNAMO,APSWS2_SPARTACUS

from traders.QuikTrader.QuikTrader import QuikTrader

bot_on_ticker = [
    {
        'ws': LWS8_SINGULARITY,
        'ws_params':{
            'start':11.08,
            'end':11.26,
            'amount_lvl': 4,
            'uh_lvl': 11.310,
            'dh_lvl': 11.030,
            'first_long': False,
            'keep_hedge':True,
            'keep_pos':False,
            'last_point':True,
            'keep_start_long':True,
            'keep_start_short':True
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
            'minute_fund':24
        },
        'dts': [
            {
                'ss':('GLDRUBF','GLH6',),
                'tfs':('M5',),
                'qs': (1,1,)
            }
        ]

    },
    {
        'ws': APSWS2_SPARTACUS,
        'ws_params':{
            'min_desc':0.1,
            'buff_0':0.01,
            'window':145,
            'smooth':None,
            'kind':'close',
            'reverse_pos':False,
            'funding': 1,
            'hour_fund':18,
            'minute_fund':24,
        },
        'dts': [
            {
                'ss':('IMOEXF','MMH6',),
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
            'minute_fund':24
        },
        'dts': [
            {
                'ss':('SBERF','SRH6',),
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
            'minute_fund':24
        },
        'dts': [
            {
                'ss':('GAZPF','GZH6',),
                'tfs':('M5',),
                'qs': (1,1,)
            }
        ]

    },

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