from traders.QuikTrader.help_funcs import *
from pprint import pprint
from time import sleep
sec_code = 'IMOEXF'
# res = get_active_order(sec_code)

res = send_transaction(sec_code,2500,'S')
print(res)
sleep(5)
res = get_order_by_trans_id(res)
pprint(res)

