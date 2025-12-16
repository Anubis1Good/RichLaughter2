import pandas as pd
from wss.WSBase import WSBase
from indicators.classic_ind import add_fractals
from indicators.pva_ind import add_nlevels_fractal

# TODO логику нужно переделывать на запоминание уровней в отдельное свойство
# проблема в том, что фрактал перекрывается и позиция закрывается
class PSWS1_(WSBase):
    """Сетка по фракталам"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'period_fractals':10,
            'amount_lvl':3,
            'min_step':0.1,
            'buff':0.02,
            'grid_dir':1,
            'offset':1
        }
        grid_dir in (None,0,1,-1)
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.period_fractals = parameters['period_fractals']
        self.amount_lvl = parameters['amount_lvl']
        self.min_step = parameters['min_step']
        self.buff = parameters['buff']
        self.offset = parameters['offset']
        if parameters['grid_dir'] == 0:
            ...
        elif parameters['grid_dir'] == 1:
            self.grid_func = self.long_grid
        elif parameters['grid_dir'] == -1:
            self.grid_func = self.short_grid
        else:
            ...
    
    def long_grid(self,row,s):
        max_pos = 1
        new_pos = None
        for idx in row.index:
            if 'top_' in idx:
                if row['close'] >= row[idx]:
                    self.need_pos[s] = 0
                    return
            if 'bot_' in idx:
                if row['close'] <= row[idx]:
                    max_pos += 1
                    new_pos = max_pos - 1   
        if self.positions[s] >= max_pos:
            new_pos = max_pos
        self.need_pos[s] = new_pos

    def short_grid(self,row,s):
        max_pos = -1
        new_pos = None
        for idx in row.index:
            if 'bot_' in idx:
                if row['close'] <= row[idx]:
                    self.need_pos[s] = 0
                    return
            if 'top_' in idx:
                if row['close'] >= row[idx]:
                    max_pos -= 1
                    new_pos = max_pos + 1
        if self.positions[s] <= max_pos:
            new_pos = max_pos
        print(row['x'],self.positions[s],max_pos,new_pos)
        self.need_pos[s] = new_pos


    def preprocessing(self, dfs, poss):
        self.last_dfs = {}
        self.update_poss_mps(poss)
        t = self.timeframes[0]
        self.last_dfs[t] = {}
        for s in dfs[t]:
            df:pd.DataFrame = dfs[t][s]
            df = add_fractals(df,self.period_fractals)
            df = add_nlevels_fractal(df,self.amount_lvl,self.min_step,self.buff,self.offset)
            self.last_dfs[t][s] = df
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]   
            self.grid_func(row,s)
        return self.need_pos