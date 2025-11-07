from traders.TraderBase import TraderBase
from traders.QuikTrader.help_funcs import get_bars,get_best_glass,get_pos_futures,close_active_order,send_transaction,get_code_orders,smart_close_active_order,get_order_by_trans_id

class QuikTrader(TraderBase):
    def __init__(self):
        super().__init__()

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