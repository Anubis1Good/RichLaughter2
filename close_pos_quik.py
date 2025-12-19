from time import sleep
from wss.help_wss.CloseWS import CloseWS
from traders.QuikTrader.QuikTrader import QuikTrader
symbols = [
    'IMOEXF',
    # 'MMH6',
    # 'CNYRUBF',
    # 'CRH6',
    # 'GAZPF',
    # 'GZH6',
    # 'SBERF',
    # 'SRH6',
    # 'GLDRUBF',
    # 'GLH6'
]
quntities = [1 for s in symbols]
bot = QuikTrader(symbols,['M5'],quntities,(CloseWS,dict()))
work = True
while work:
    bot.run()
    if bot.ws.all_close:
        work = False
    sleep(15)
print('All Close!!')