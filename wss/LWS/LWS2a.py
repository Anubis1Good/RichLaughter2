import pandas as pd
from wss.WSBase import WSBase
from indicators.classic_ind import add_rsi,add_atr
    
class LWS8_SINGULARITY(WSBase):
    """парный реверс грид-бот c хеджем"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'start':2500,
            'end':3000,
            'amount_lvl': 5,
            'uh_lvl': 3100,
            'dh_lvl': 2400,
            'first_long': False,
            'keep_hedge':True,
            'keep_pos':False
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.max_pos = parameters['amount_lvl']
        self.hedge_pos = parameters['amount_lvl'] - 1
        delta_se = parameters['end'] - parameters['start']
        step_lvl = delta_se / (parameters['amount_lvl'] - 1)
        self.lvls = [parameters['start'] + step_lvl*i for i in range(parameters['amount_lvl'])]
        print(symbols,self.lvls)
        self.uh_lvl = parameters['uh_lvl']
        self.dh_lvl = parameters['dh_lvl']
        self.first_long = parameters['first_long']
        self.keep_hedge = parameters['keep_hedge']
        self.keep_pos = parameters.get('keep_pos',False)
        self.middle_lvl = sum(self.lvls) / len(self.lvls)
        self.in_work = True
    
    def get_need_pos(self,pos_data):
        new_pos_long,new_pos_short,max_pos_long,max_pos_short = pos_data
        s_l,s_s = (self.symbols[0],self.symbols[1]) if self.first_long else (self.symbols[1],self.symbols[0])
        cur_pos_l = self.positions[s_l]
        cur_pos_s = self.positions[s_s]
        need_pos = {}
        if cur_pos_l >= max_pos_long:
            new_pos_long = max_pos_long
        if cur_pos_s <= max_pos_short:
            new_pos_short = max_pos_short
        if self.keep_pos:
            if new_pos_long != 0:
                if cur_pos_l > new_pos_long:
                    new_pos_long = None
            if new_pos_short != 0:
                if cur_pos_s < new_pos_short:
                    new_pos_short = None
        need_pos[s_l] = new_pos_long
        need_pos[s_s] = new_pos_short
        return need_pos
    
    def get_pos_on_grid(self,row):
        if self.dh_lvl:
            if row['close'] < self.dh_lvl:
                self.in_work = False if self.keep_hedge else True
                return (self.hedge_pos,-self.hedge_pos,self.max_pos,-self.max_pos)
        if self.uh_lvl:
            if row['close'] > self.uh_lvl:
                self.in_work = False if self.keep_hedge else True
                return (self.hedge_pos,-self.hedge_pos,self.max_pos,-self.max_pos)
        new_pos_long,new_pos_short = None,None
        max_pos_long,max_pos_short = 0,0
        for lvl in self.lvls:
            if row['close'] <= lvl:
                max_pos_long += 1
                new_pos_long = max_pos_long - 1
            if row['close'] >= lvl:
                max_pos_short -= 1
                new_pos_short = max_pos_short + 1
        return new_pos_long,new_pos_short,max_pos_long,max_pos_short
    
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            self.last_dfs[tf1][s] = df
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        if self.in_work:
            tf1 = self.timeframes[0]
            s1 = self.symbols[0]
            row = self.last_dfs[tf1][s1].iloc[-1]
            pos_data = self.get_pos_on_grid(row)
            self.need_pos = self.get_need_pos(pos_data)
        else:
            self.need_pos = {s: None for s in self.symbols}
        return self.need_pos
    
