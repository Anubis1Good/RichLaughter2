import pandas as pd
from typing import Dict, List, Any, Optional, Union
from traders.TraderBase import TraderBase
from traders.QuikTrader.help_funcs import get_bars,get_best_glass,get_pos_futures,close_active_order,send_transaction,get_code_orders,smart_close_active_order,get_order_by_trans_id
from wss.WSBase import WSBase

class QuikTrader(TraderBase):
    def __init__(self,symbols: Union[str, List[str]] = "IMOEXF",class_code='SPBFUT',granularity='M5',quantity = 1,ws:tuple=(WSBase,(20,)),need_debug=False,smart_reset=True):
        super().__init__()


    def start_info(self):
        print('QuikTrader')

    def _get_balance(self):
        pass
    
    def _check_risk(self):
        pass

    def _check_time(self):
        pass

    def _check_position(self):
        pass

    def _check_req(self):
        pass

    def _send_open(self,direction,quantity):
        pass

    def _send_close(self,direction,quantity):
        pass

    def _reverse_pos(self,direction):
        pass
    
    def _close_all_pos(self):
        pass

    def _reset_req(self):
        pass

    def _get_df(self) -> pd.DataFrame:
        df = pd.DataFrame()
        return df
    
    def _debug_log(self):
        pass
    
    def _work_ws(self,new_pos,pos):
        pass

    def run(self):
        pass

    def _check_pos_on_orders(self):
        orders = get_code_orders(self.sec_code)
        pos = self.start_pos
        for order in orders:
            flags = bin(order['flags'])
            if flags[-1] == '0' and flags[-2] == '0':
                delta = order['qty']
            else:
                delta = order['qty'] - order['balance']
            if self._check_today(order):
                if flags[-3] == '1':
                    pos -= delta
                else:
                    pos += delta
        return int(pos)