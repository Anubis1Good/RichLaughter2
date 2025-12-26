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
    
class OpenWSCondition(WSBase):
    """close choosen position"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        """
        parameters = {
            'need_pos_up':None,
            'need_pos_down':1,
            'condition_up': None,
            'condition_down': 2000
        }
        """
        self.need_pos_up = parameters['need_pos_up']
        self.need_pos_down = parameters['need_pos_down']
        self.condition_up = parameters['condition_up']
        self.condition_down = parameters['condition_down']
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        self.last_dfs = dfs
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]
            new_pos = None
            if self.condition_up:
                if row['close'] >= self.condition_up:
                    new_pos = self.need_pos_up
            if self.condition_down:
                if row['close'] <= self.condition_down:
                    new_pos = self.need_pos_down
            self.need_pos[s] = new_pos

        return self.need_pos