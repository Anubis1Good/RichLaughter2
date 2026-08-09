from wss.LWS.LWS2a import LWS8_SINGULARITY,LWS8_LITE
from wss.APSWS.APSWS1a import APSWS1_DYNAMO,APSWS2_SPARTACUS

from traders.QuikTrader.QuikTrader import QuikTrader

bot_on_ticker = [
    # {
    #     'ws': LWS8_SINGULARITY,
    #     'ws_params':{
    #         'start':11.268,
    #         'end':11.584,
    #         'amount_lvl': 4,
    #         'uh_lvl': 11.610,
    #         'dh_lvl': 11.230,
    #         'first_long': False,
    #         'keep_hedge':True,
    #         'keep_pos':False,
    #         'last_point':True,
    #         'keep_start_long':True,
    #         'keep_start_short':True
    #     },
    #     'dts': [
    #         {
    #             'ss':('CNYRUBF','CRH6',),
    #             'tfs':('M5',),
    #             'qs': (1,1,)
    #         }
    #     ]

    # },
    # {
    #     'ws': LWS8_LITE,
    #     'ws_params':{
    #         'start_lvl':11355,
    #         'end_lvl':11400,
    #         'uh_lvl': 11415,
    #         'dh_lvl': 11340,
    #         'first_long': False,
    #         'keep_hedge':True,
    #         'last_point':False,
    #         'keep_start_long':False,
    #         'keep_start_short':False,
    #         'close_hedge':False,
    #         'unfreeze':True,
    #     },
    #     'dts': [
    #         {
    #             'ss':('GLDRUBF','GLH6',),
    #             'tfs':('M5',),
    #             'qs': (1,1,)
    #         }
    #     ]

    # },
    # {
    #     'ws': LWS8_LITE,
    #     'ws_params':{
    #         'start_lvl':2736,
    #         'end_lvl':2760,
    #         'uh_lvl': 2765.5,
    #         'dh_lvl': 2730,
    #         'first_long': False,
    #         'keep_hedge':True,
    #         'last_point':False,
    #         'keep_start_long':False,
    #         'keep_start_short':False,
    #         'close_hedge':False,
    #         'unfreeze':True,
    #     },
    #     'dts': [
    #         {
    #             'ss':('IMOEXF','MMH6',),
    #             'tfs':('M5',),
    #             'qs': (1,1,)
    #         }
    #     ]

    # },
    {
        'ws': LWS8_SINGULARITY,
        'ws_params':{
            'start':121,
            'end':129,
            'amount_lvl': 4,
            'uh_lvl': 130,
            'dh_lvl': 120,
            'first_long': False,
            'keep_hedge':True,
            'keep_pos':False,
            'last_point':True,
            'keep_start_long':False,
            'keep_start_short':False
        },
        'dts': [
            {
                'ss':('GAZPF','GZH6',),
                'tfs':('M5',),
                'qs': (1,1,)
            }
        ]

    },
    # {
    #     'ws': APSWS1_DYNAMO,
    #     'ws_params':{
    #         'first_long': True,
    #         'funding': True,
    #         'hour_fund':18,
    #         'minute_fund':24
    #     },
    #     'dts': [
    #         {
    #             'ss':('SBERF','SRH6',),
    #             'tfs':('M5',),
    #             'qs': (1,1,)
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