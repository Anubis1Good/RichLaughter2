import pandas as pd
from wss.WSBase import WSBase


class CloseWS(WSBase):
    """close choosen position"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.all_close = False
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        self.last_dfs = dfs
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        self.all_close = True
        for s in self.positions:
            if self.positions[s] != 0:
                self.all_close = False
                self.need_pos[s] = 0
        return self.need_pos