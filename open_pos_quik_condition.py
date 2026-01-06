from time import sleep
from wss.help_wss.OpenWS import OpenWSCondition
from traders.QuikTrader.QuikTrader import QuikTrader
confs = [
    {
        'symbols':['CNYRUBF',],
        'quntities':[1,],
        'params':{
            'need_pos_up':-4,
            'need_pos_down':None,
            'condition_up': 11.350,
            'condition_down': None
        }
    },
    # {
    #     'symbols':['CRH6',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':None,
    #         'need_pos_down':None,
    #         'condition_up': None,
    #         'condition_down': None
    #     }
    # },
    # {
    #     'symbols':['IMOEXF',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':-1,
    #         'need_pos_down':None,
    #         'condition_up': 2753,
    #         'condition_down': None
    #     }
    # },
    # {
    #     'symbols':['MMH6',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':0,
    #         'need_pos_down':2,
    #         'condition_up': 2812,
    #         'condition_down': 2787
    #     }
    # },
    # {
    #     'symbols':['RMH6',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':None,
    #         'need_pos_down':0,
    #         'condition_up': None,
    #         'condition_down': 1118
    #     }
    # },
    # {
    #     'symbols':['GAZPF',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':None,
    #         'need_pos_down':0,
    #         'condition_up': None,
    #         'condition_down': 124.75
    #     }
    # },
    # {
    #     'symbols':['GZH6',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':-1,
    #         'need_pos_down':None,
    #         'condition_up': 13040,
    #         'condition_down': None
    #     }
    # },
    # {
    #     'symbols':['SBERF',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':None,
    #         'need_pos_down':None,
    #         'condition_up': None,
    #         'condition_down': None
    #     }
    # },
    # {
    #     'symbols':['SRH6',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':None,
    #         'need_pos_down':None,
    #         'condition_up': None,
    #         'condition_down': None
    #     }
    # },
    # {
    #     'symbols':['GLDRUBF',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':None,
    #         'need_pos_down':None,
    #         'condition_up': None,
    #         'condition_down': None
    #     }
    # },
    # {
    #     'symbols':['GLH6',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':None,
    #         'need_pos_down':None,
    #         'condition_up': None,
    #         'condition_down': None
    #     }
    # },
    # {
    #     'symbols':['BRF6',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':0,
    #         'need_pos_down':None,
    #         'condition_up': 61.69,
    #         'condition_down': None
    #     }
    # },
    # {
    #     'symbols':['NGZ5',],
    #     'quntities':[1,],
    #     'params':{
    #         'need_pos_up':None,
    #         'need_pos_down':None,
    #         'condition_up': None,
    #         'condition_down': None
    #     }
    # },
]
bots = []
for conf in confs:
    print(conf['symbols'],conf['params'])
    bot = QuikTrader(conf['symbols'],['M5'],conf['quntities'],(OpenWSCondition,conf['params']),close_on_time=False)
    bots.append(bot)
print('Run OpenQuikCondition')
work = True
while work:
    for bot in bots:
        bot.run()
    sleep(15)
