from wss.WSBase import WSBase


class OpenWS(WSBase):
    """close choosen position"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        """
        parameters = {
            'need_pos':1
        }
        """
        self.need_pos_inner = { s: parameters['need_pos'] for s in self.symbols}
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        self.last_dfs = dfs
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        self.need_pos = self.need_pos_inner.copy()
        return self.need_pos