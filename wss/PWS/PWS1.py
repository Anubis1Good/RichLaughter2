import pandas as pd
from wss.WSBase import WSBase
from indicators.classic_ind import add_donchan_channel,add_atr


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
    """парный грид-бот по DC"""
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

class PWS2_DIRDC(WSBase):
    "направленный DDC"
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        #enter = 0-center 1-edge 
        """
        parameters = {
            'period':50,
            'dir': 0,
            'enter': 1,
            'defense': True
        }
        """
        self.period = parameters['period']
        self.dir = parameters['dir']
        self.enter = parameters['enter']
        self.defense = parameters['defense']
    
    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        self.update_poss_mps(poss)
        for t in dfs:
            for s in dfs[t]:
                df:pd.DataFrame = dfs[t][s]
                df = add_donchan_channel(df,self.period)
        return self.last_dfs

    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]   
            if self.dir == 1:
                enter = 'average' if self.enter == 0 else 'min_hb'
                if row['low'] <= row[enter]:
                    self.need_pos[s] = 1 if self.defense else 0
                elif row['high'] >= row['max_hb']:
                    self.need_pos[s] = 0 if self.defense else 1
            elif self.dir == -1:
                enter = 'average' if self.enter == 0 else 'max_hb'
                if row['low'] <= row['min_hb']:
                    self.need_pos[s] = 0 if self.defense else -1
                elif row['high'] >= row[enter]:
                    self.need_pos[s] = -1 if self.defense else 0
            else:
                if row['low'] <= row['min_hb']:
                    self.need_pos[s] = 1 if self.defense else -1
                elif row['high'] >= row['max_hb']:
                    self.need_pos[s] = -1 if self.defense else 1
        return self.need_pos

# TODO есть какие-то проблемы
class PWS2_DIRATR(WSBase):
    "направленный ATR"
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        """
        parameters = {
            'period':5,
            'dir': 0,
            'n_atr': 3,
            'defense': True
        }
        """
        self.period = parameters['period']
        self.dir = parameters['dir']
        self.n_atr = parameters['n_atr']
        self.defense = parameters['defense']
    
    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        self.update_poss_mps(poss)
        for t in dfs:
            for s in dfs[t]:
                df:pd.DataFrame = dfs[t][s]
                df = add_atr(df,self.period)
                df['prev_close'] = df['close'].shift(1)
                df['top_line'] = df['prev_close'] + df['atr'] * self.n_atr
                df['bottom_line'] = df['prev_close'] - df['atr'] * self.n_atr
        return self.last_dfs

    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]   
            if self.dir == 1:
                if row['close'] <= row['bottom_line']:
                    self.need_pos[s] = 1 if self.defense else 0
                elif row['close'] >= row['top_line']:
                    self.need_pos[s] = 0 if self.defense else 1
            elif self.dir == -1:
                if row['close'] <= row['bottom_line']:
                    self.need_pos[s] = 0 if self.defense else -1
                elif row['close'] >= row['top_line']:
                    self.need_pos[s] = -1 if self.defense else 0
            else:
                if row['close'] <= row['bottom_line']:
                    self.need_pos[s] = 1 if self.defense else -1
                elif row['close'] >= row['top_line']:
                    self.need_pos[s] = -1 if self.defense else 1
        return self.need_pos