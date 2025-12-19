from time import sleep
from wss.help_wss.OpenWS import OpenWS
from traders.QuikTrader.QuikTrader import QuikTrader
confs = [
    # {
    #     'symbols':['CNYRUBF',],
    #     'quntities':[1,],
    #     'params':{'need_pos':1}
    # },
    # {
    #     'symbols':['CRH6',],
    #     'quntities':[1,],
    #     'params':{'need_pos':-1}
    # },
    {
        'symbols':['IMOEXF',],
        'quntities':[1,],
        'params':{'need_pos':1}
    },
    {
        'symbols':['MMH6',],
        'quntities':[1,],
        'params':{'need_pos':-1}
    },
]
bots = []
for conf in confs:
    bot = QuikTrader(conf['symbols'],['M5'],conf['quntities'],(OpenWS,conf['params']))
    bots.append(bot)
work = True
while work:
    for bot in bots:
        bot.run()
    sleep(15)
