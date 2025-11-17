import pandas as pd
from wss.WSBase import WSBase
from indicators.classic_ind import add_donchan_channel


#Исправить дребезжание
class PWS1_GRIDC(WSBase):
    """грид-бот по DC"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'period':50,
            'amount_lvl': 5,
            'grid_dir': 1,
            'per_limit': 0.1,
            'keep': False
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.period = parameters['period']
        self.amount_lvl = parameters['amount_lvl'] + 1
        self.range_amount = range(self.amount_lvl)
        self.grid_dir = parameters['grid_dir']
        self.per_limit = parameters['per_limit']
        self.keep = parameters['keep']
        if self.grid_dir == 1: #long
            self.grid_func = self.long_grid
        elif self.grid_dir == -1:
            self.grid_func = self.short_grid
        else:
            self.grid_func = self.neutral_grid
    
    def long_grid(self,row,s):
        if row['allow_grid']:
            new_pos = -1
            for i in self.range_amount:
                if row['close'] <= row['lvl_'+str(i)]:
                    new_pos += 1
                elif row['close'] <= row['lvl_'+str(i)] + row['buff']:
                    new_pos = None
                    break
            if new_pos == 0:
                new_pos = None
            elif new_pos == -1:
                if self.keep:
                    new_pos = None
                else:
                    new_pos = 0
        else:
            new_pos = None
        self.need_pos[s] = new_pos

    
    def short_grid(self,row,s):
        if row['allow_grid']:
            new_pos = 1
            for i in self.range_amount:
                if row['close'] >= row['lvl_'+str(i)]:
                    new_pos -= 1
                elif row['close'] >= row['lvl_'+str(i)] - row['buff']:
                    new_pos = None
                    break
            if new_pos == 0:
                new_pos = None
            elif new_pos == 1:
                if self.keep:
                    new_pos = None
                else:
                    new_pos = 0
        else:
            new_pos = None
        self.need_pos[s] = new_pos
    
    def neutral_grid(self,row,s):
        if row['allow_grid']:
            new_pos = 0
            for i in self.range_amount:
                if row['lvl_'+str(i)] > row['average']:
                    if row['close'] >= row['lvl_'+str(i)]:
                        new_pos -= 1
                    elif row['close'] >= row['lvl_'+str(i)] - row['buff'] and self.positions[s] > 0:
                        new_pos = None
                        break
                elif row['lvl_'+str(i)] < row['average']:
                    if row['close'] <= row['lvl_'+str(i)]:
                        new_pos += 1
                    elif row['close'] <= row['lvl_'+str(i)] + row['buff'] and self.positions[s] < 0:
                        new_pos = None
                        break
            if new_pos == 0:
                new_pos = None
        else:
            new_pos = None
        self.need_pos[s] = new_pos

    
    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        self.update_poss_mps(poss)
        for t in dfs:
            for s in dfs[t]:
                df:pd.DataFrame = dfs[t][s]
                df = add_donchan_channel(df,self.period)
                df['dcr'] = df['max_hb'] - df['min_hb']
                df['step'] = df['dcr'] / self.amount_lvl
                df['buff'] = df['step'] / 2
                df['per_step'] = (df['step'] / df['close']) * 100
                df['allow_grid'] = df['per_step'] > self.per_limit
                for i in self.range_amount:
                    df['lvl_'+str(i)] = df['min_hb'] + df['step'] * i
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]   
            self.grid_func(row,s)
        return self.need_pos
    
class PWS1_PRGDC(WSBase):
    """грид-бот по DC"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'period':50,
            'amount_lvl': 5,
            'per_limit': 0.1,
            'keep': False
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.period = parameters['period']
        self.amount_lvl = parameters['amount_lvl'] + 1
        self.range_amount = range(self.amount_lvl)
        self.grid_dirs = {s: 1 for s in self.symbols}
        self.grid_dirs[self.symbols[1]] = -1
        self.per_limit = parameters['per_limit']
        self.keep = parameters['keep']

    def grid_func(self,row,s):
        if self.grid_dirs[s] == 1: #long
            self.long_grid(row,s)
        else:
            self.short_grid(row,s)

    def long_grid(self,row,s):
        if row['allow_grid']:
            new_pos = -1
            for i in self.range_amount:
                if row['close'] <= row['lvl_'+str(i)]:
                    new_pos += 1
                elif row['close'] <= row['lvl_'+str(i)] + row['buff']:
                    new_pos = None
                    break
            if new_pos == 0:
                new_pos = None
            elif new_pos == -1:
                if self.keep:
                    new_pos = None
                else:
                    new_pos = 0
        else:
            new_pos = None
        self.need_pos[s] = new_pos

    
    def short_grid(self,row,s):
        if row['allow_grid']:
            new_pos = 1
            for i in self.range_amount:
                if row['close'] >= row['lvl_'+str(i)]:
                    new_pos -= 1
                elif row['close'] >= row['lvl_'+str(i)] - row['buff']:
                    new_pos = None
                    break
            if new_pos == 0:
                new_pos = None
            elif new_pos == 1:
                if self.keep:
                    new_pos = None
                else:
                    new_pos = 0
        else:
            new_pos = None
        self.need_pos[s] = new_pos

    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        self.update_poss_mps(poss)
        for t in dfs:
            for s in dfs[t]:
                df:pd.DataFrame = dfs[t][s]
                df = add_donchan_channel(df,self.period)
                df['dcr'] = df['max_hb'] - df['min_hb']
                df['step'] = df['dcr'] / self.amount_lvl
                df['buff'] = df['step'] / 2
                df['per_step'] = (df['step'] / df['close']) * 100
                df['allow_grid'] = df['per_step'] > self.per_limit
                for i in self.range_amount:
                    df['lvl_'+str(i)] = df['min_hb'] + df['step'] * i
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]   
            self.grid_func(row,s)
        return self.need_pos
