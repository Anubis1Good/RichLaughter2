from time import sleep
from wss.help_wss.OpenWS import OpenWS
from traders.QuikTrader.QuikTrader import QuikTrader
confs = [
    # {
    #     'symbols':['CNYRUBF',],
    #     'quntities':[1,],
    #     'params':{'need_pos':-3}
    # },
    # {
    #     'symbols':['CRH6',],
    #     'quntities':[1,],
    #     'params':{'need_pos':3}
    # },
    # {
    #     'symbols':['IMOEXF',],
    #     'quntities':[1,],
    #     'params':{'need_pos':-3}
    # },
    # {
    #     'symbols':['MMH6',],
    #     'quntities':[1,],
    #     'params':{'need_pos':1}
    # },
    # {
    #     'symbols':['GAZPF',],
    #     'quntities':[1,],
    #     'params':{'need_pos':-2}
    # },
    # {
    #     'symbols':['GZH6',],
    #     'quntities':[1,],
    #     'params':{'need_pos':-2}
    # },
    # {
    #     'symbols':['SBERF',],
    #     'quntities':[1,],
    #     'params':{'need_pos':-2}
    # },
    # {
    #     'symbols':['SRH6',],
    #     'quntities':[1,],
    #     'params':{'need_pos':-2}
    # },
    # {
    #     'symbols':['GLDRUBF',],
    #     'quntities':[1,],
    #     'params':{'need_pos':-2}
    # },
    # {
    #     'symbols':['GLH6',],
    #     'quntities':[1,],
    #     'params':{'need_pos':2}
    # },
    # {
    #     'symbols':['BRF6',],
    #     'quntities':[1,],
    #     'params':{'need_pos':0}
    # },
]
bots = []
for conf in confs:
    bot = QuikTrader(conf['symbols'],['M5'],conf['quntities'],(OpenWS,conf['params']),close_on_time=False)
    bots.append(bot)
print('Run OpenQuik')
work = True
while work:
    for bot in bots:
        bot.run()
    sleep(15)
