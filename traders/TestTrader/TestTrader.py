import pandas as pd
from typing import Dict, List, Any, Optional, Union
from traders.TraderBase import TraderBase

class TestTrader(TraderBase):
    def __init__(self, symbols, timeframes, quantity, ws = ..., sufix_debug='Test1', need_debug=False, charts:Dict[str,str]=dict(), fee=0.0002,close_on_time=False,close_map=('23:30','23:30','23:30','23:30','23:30','17:50','17:50')):
        """charts = {
        'tf1': {
            's1':'./df1.csv',
            's2':'./df2.csv'
            }
        }"""
        super().__init__(symbols, timeframes, quantity, ws, sufix_debug, need_debug)
        self.charts = charts
        self.trade_data = {
            symbol : {
                'total':0,
                'count':0,
                'fees': 0,
                'total_wfees_per':0,
                'equity':[0],
                'equity_fee':[0],
                'pos':0,
                'mp':0,
                'o_longs':[],
                'o_shorts':[],
                'c_longs':[],
                'c_shorts':[]
            } for symbol in self.symbols
        }
        self.fee_one_p = (fee / 2) * 100
        self.open_fee = {symbol:0 for symbol in self.symbols}
        self.close_on_time = close_on_time
        if self.close_on_time:
            ...
        self.close_map = close_map
