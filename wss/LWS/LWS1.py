import pandas as pd
from wss.WSBase import WSBase
from indicators.classic_ind import add_rsi,add_atr

class LWS1_FIRSTGRID(WSBase):
    """грид-бот"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'lvls':(200,300,400,500),
            'us_lvl': None,
            'ds_lvl': 100,
            'grid_dir': 1
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.lvls = parameters['lvls']
        self.buffs = []
        buff = 0
        for i,lvl in enumerate(self.lvls):
            if i != len(self.lvls) - 1:
                buff = (self.lvls[i+1] - lvl) / 2
            self.buffs.append(buff)
        self.us_lvl = parameters['us_lvl']
        self.ds_lvl = parameters['ds_lvl']
        self.grid_dir = parameters['grid_dir']
        self.middle_lvl = sum(self.lvls) / len(self.lvls)
        self.in_work = True
        if self.grid_dir == 1: #long
            self.grid_func = self.long_grid
        elif self.grid_dir == -1:
            self.grid_func = self.short_grid
        else:
            self.grid_func = self.neutral_grid
    
    def long_grid(self,row,s):
        if self.ds_lvl:
            if row['close'] < self.ds_lvl:
                self.in_work = False
                return False    
        if self.us_lvl:
            if row['close'] > self.us_lvl:
                self.need_pos[s] = 0
                return True
        new_pos = 0
        for i,lvl in enumerate(self.lvls):
            if row['close'] <= lvl:
                new_pos += 1
            elif lvl + self.buffs[i] >= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        self.need_pos[s] = new_pos
        return True
    
    def short_grid(self,row,s):
        if self.us_lvl:
            if row['close'] > self.us_lvl:
                self.in_work = False
                return False    
        if self.ds_lvl:
            if row['close'] < self.ds_lvl:
                self.need_pos[s] = 0
                return True
        new_pos = 0
        for i,lvl in enumerate(self.lvls):
            if row['close'] >= lvl:
                new_pos -= 1
            elif lvl - self.buffs[i] <= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        self.need_pos[s] = new_pos
        return True
    
    def neutral_grid(self,row,s):
        if self.us_lvl:
            if row['close'] > self.us_lvl:
                self.in_work = False
                return False    
        if self.ds_lvl:
            if row['close'] < self.ds_lvl:
                self.in_work = False
                return False 
        new_pos = 0
        for i,lvl in enumerate(self.lvls):
            if lvl < self.middle_lvl:
                if row['close'] <= lvl:
                    new_pos += 1
                elif lvl + self.buffs[i] >= row['close'] and self.positions[s] > 0:
                    new_pos = None
                    break
            elif lvl > self.middle_lvl:
                if row['close'] >= lvl:
                    new_pos -= 1
                elif lvl - self.buffs[i] <= row['close'] and self.positions[s] < 0:
                    new_pos = None
                    break
        if new_pos == 0:
            new_pos = None
        self.need_pos[s] = new_pos
        return True
    
    def preprocessing(self, dfs, poss):
        super().preprocessing(dfs, poss)
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        if self.in_work:
            tf1 = self.timeframes[0]
            for s in self.last_dfs[tf1]:
                row = self.last_dfs[tf1][s].iloc[-1]
                if not self.grid_func(row,s):
                    break
        else:
            self.need_pos = {s: 0 for s in self.symbols}
        return self.need_pos
    
class LWS1_AUTOGRID(WSBase):
    """грид-бот c автоматическим рассчетом уровней"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'start':200,
            'end':300,
            'amount_lvl': 5,
            'us_lvl': None,
            'ds_lvl': 100,
            'grid_dir': 1,
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        step_lvl = (parameters['end'] - parameters['start']) / parameters['amount_lvl']
        self.buff = step_lvl / 2
        self.grid_dir = parameters['grid_dir']
        if self.grid_dir == 1: #long
            self.grid_func = self.long_grid
            offset = 0
        elif self.grid_dir == -1:
            self.grid_func = self.short_grid
            offset = step_lvl
        else:
            self.grid_func = self.neutral_grid
            offset = step_lvl / 2

        self.lvls = [parameters['start'] + step_lvl*i + offset for i in range(parameters['amount_lvl'])]
        print(self.symbols,'lvls:',self.lvls)
        self.us_lvl = parameters['us_lvl']
        self.ds_lvl = parameters['ds_lvl']
        self.middle_lvl = sum(self.lvls) / len(self.lvls)
        self.in_work = True

    def long_grid(self,row,s):
        if self.ds_lvl:
            if row['close'] < self.ds_lvl:
                self.in_work = False
                return False    
        if self.us_lvl:
            if row['close'] > self.us_lvl:
                self.need_pos[s] = 0
                return True
        new_pos = 0
        for lvl in self.lvls:
            if row['close'] <= lvl:
                new_pos += 1
            elif lvl + self.buff >= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        self.need_pos[s] = new_pos
        return True
    
    def short_grid(self,row,s):
        if self.us_lvl:
            if row['close'] > self.us_lvl:
                self.in_work = False
                return False    
        if self.ds_lvl:
            if row['close'] < self.ds_lvl:
                self.need_pos[s] = 0
                return True
        new_pos = 0
        for lvl in self.lvls:
            if row['close'] >= lvl:
                new_pos -= 1
            elif lvl - self.buff <= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        self.need_pos[s] = new_pos
        return True
    
    def neutral_grid(self,row,s):
        if self.us_lvl:
            if row['close'] > self.us_lvl:
                self.in_work = False
                return False    
        if self.ds_lvl:
            if row['close'] < self.ds_lvl:
                self.in_work = False
                return False 
        new_pos = 0
        for lvl in self.lvls:
            if lvl < self.middle_lvl:
                if row['close'] <= lvl:
                    new_pos += 1
                elif lvl + self.buff >= row['close'] and self.positions[s] > 0:
                    new_pos = None
                    break
            elif lvl > self.middle_lvl:
                if row['close'] >= lvl:
                    new_pos -= 1
                elif lvl - self.buff <= row['close'] and self.positions[s] < 0:
                    new_pos = None
                    break
        if new_pos == 0:
            new_pos = None
        self.need_pos[s] = new_pos
        return True
    
    def preprocessing(self, dfs, poss):
        super().preprocessing(dfs, poss)
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        if self.in_work:
            tf1 = self.timeframes[0]
            for s in self.last_dfs[tf1]:
                row = self.last_dfs[tf1][s].iloc[-1]
                if not self.grid_func(row,s):
                    break
        else:
            self.need_pos = {s: 0 for s in self.symbols}
        return self.need_pos
    
class LWS2_SWIMGRID(WSBase):
    """плавающий грид-бот c автоматическим рассчетом уровней"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'amount_lvl': 5,
            'per_step':0.1,
            'grid_dir': 1,
            'keep':False,
            'reset_n':2
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.amount_lvl = parameters['amount_lvl']
        self.first_run = {s: True for s in self.symbols}
        self.step = {s: None for s in self.symbols}
        self.lvls = {s: list() for s in self.symbols}
        self.per_step = parameters['per_step']
        self.buff = self.per_step / 2
        self.grid_dir = parameters['grid_dir']
        self.keep = parameters['keep']
        self.reset_n = parameters['reset_n']
        self.up_lvls = {s: None for s in self.symbols}
        self.down_lvls = {s: None for s in self.symbols}
        self.middle_lvls = {s: None for s in self.symbols}
        if self.grid_dir == 1: #long
            self.grid_func = self.long_grid
        elif self.grid_dir == -1:
            self.grid_func = self.short_grid
        else:
            self.grid_func = self.neutral_grid

    def update_grid(self,s,row):
        self.step[s] = (row['close'] / 100) * self.per_step
        if self.grid_dir == 1: #long
            self.long_init_grid(s,row)
        elif self.grid_dir == -1:
            self.short_init_grid(s,row)
        else:
            self.neutral_init_grid(s,row)

    def long_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        start_lvl = self.step[s] * n_lvl
        for i in range(self.amount_lvl):
            lvl = start_lvl - self.step[s]*i
            self.lvls[s].append(lvl)
        self.up_lvls[s] = start_lvl + self.step[s] * self.reset_n
        self.down_lvls[s] = start_lvl - self.step[s] * (self.amount_lvl + self.reset_n - 1)

    def short_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        start_lvl = self.step[s] * n_lvl
        for i in range(self.amount_lvl):
            lvl = start_lvl + self.step[s]*i
            self.lvls[s].append(lvl)
        self.up_lvls[s] = start_lvl + self.step[s] * (self.amount_lvl + self.reset_n - 1)
        self.down_lvls[s] = start_lvl - self.step[s] * self.reset_n

    def neutral_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        middle_lvl = self.step[s] * n_lvl
        for i in range(1,self.amount_lvl // 2+1):
            lvl = middle_lvl + self.step[s]*i
            self.lvls[s].append(lvl)
            lvl = middle_lvl - self.step[s]*i
            self.lvls[s].append(lvl)
        self.lvls[s].append(middle_lvl)
        self.lvls[s].sort()
        self.up_lvls[s] = max(self.lvls[s]) + self.step[s] * self.reset_n
        self.down_lvls[s] = min(self.lvls[s]) - self.step[s] * self.reset_n
        self.middle_lvls[s] = middle_lvl

    def long_grid(self,row,s):
        new_pos = -1
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if row['close'] <= lvl:
                new_pos += 1
            elif lvl + buff >= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        elif new_pos == -1:
            if self.keep:
                new_pos = None
            else:
                new_pos = 0
        self.need_pos[s] = new_pos

    
    def short_grid(self,row,s):
        new_pos = 1
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if row['close'] >= lvl:
                new_pos -= 1
            elif lvl - buff <= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        elif new_pos == 1:
            if self.keep:
                new_pos = None
            else:
                new_pos = 0
        self.need_pos[s] = new_pos
    
    def neutral_grid(self,row,s):
        new_pos = 0
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if lvl < self.middle_lvls[s]:
                if row['close'] <= lvl:
                    new_pos += 1
                elif lvl + buff >= row['close'] and self.positions[s] > 0:
                    new_pos = None
                    break
            elif lvl > self.middle_lvls[s]:
                if row['close'] >= lvl:
                    new_pos -= 1
                elif lvl - buff <= row['close'] and self.positions[s] < 0:
                    new_pos = None
                    break
        if new_pos == 0:
            new_pos = None
        self.need_pos[s] = new_pos
        
    
    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]
            if self.first_run[s]:
                self.first_run[s] = False
                self.update_grid(s,row)
                print('LWS2_SWIMGRID:',s,self.lvls[s],self.up_lvls[s],self.down_lvls[s])
            if row['close'] > self.up_lvls[s] or row['close'] < self.down_lvls[s]:
                self.update_grid(s,row)
            self.grid_func(row,s)
            # print(self.lvls[s],row['close'],self.need_pos[s])
        return self.need_pos
    
class LWS2_SWIMIGSON(WSBase):
    """ИСП1 плавающий грид-бот c автоматическим рассчетом уровней"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'amount_lvl': 5,
            'per_step':0.1,
            'grid_dir': 1,
            'keep':False,
            'reset_n':2
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.amount_lvl = parameters['amount_lvl']
        self.first_run = {s: True for s in self.symbols}
        self.step = {s: None for s in self.symbols}
        self.lvls = {s: list() for s in self.symbols}
        self.per_step = parameters['per_step']
        self.buff = self.per_step / 2
        self.grid_dir = parameters['grid_dir']
        self.keep = parameters['keep']
        self.reset_n = parameters['reset_n']
        self.up_lvls = {s: None for s in self.symbols}
        self.down_lvls = {s: None for s in self.symbols}
        self.middle_lvls = {s: None for s in self.symbols}
        if self.grid_dir == 1: #long
            self.grid_func = self.long_grid
        elif self.grid_dir == -1:
            self.grid_func = self.short_grid
        else:
            self.grid_func = self.neutral_grid

    def init_grid(self,s,row):
        self.step[s] = (row['close'] / 100) * self.per_step
        if self.grid_dir == 1: #long
            self.long_init_grid(s,row)
        elif self.grid_dir == -1:
            self.short_init_grid(s,row)
        else:
            self.neutral_init_grid(s,row)
    
    def update_grid(self,s,row):
        if row['close'] > self.up_lvls[s]:
            self.step[s] = (row['close'] / 100) * self.per_step
            if self.grid_dir != 0:
                self.long_init_grid(s,row)
            else:
                self.neutral_init_grid(s,row)
        elif row['close'] < self.down_lvls[s]:
            self.step[s] = (row['close'] / 100) * self.per_step
            if self.grid_dir != 0:
                self.short_init_grid(s,row)
            else:
                self.neutral_init_grid(s,row)


    def long_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        start_lvl = self.step[s] * n_lvl
        for i in range(self.amount_lvl):
            lvl = start_lvl - self.step[s]*i
            self.lvls[s].append(lvl)
        self.up_lvls[s] = start_lvl + self.step[s] * self.reset_n
        self.down_lvls[s] = start_lvl - self.step[s] * (self.amount_lvl + self.reset_n - 1)

    def short_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        start_lvl = self.step[s] * n_lvl
        for i in range(self.amount_lvl):
            lvl = start_lvl + self.step[s]*i
            self.lvls[s].append(lvl)
        self.up_lvls[s] = start_lvl + self.step[s] * (self.amount_lvl + self.reset_n - 1)
        self.down_lvls[s] = start_lvl - self.step[s] * self.reset_n

    def neutral_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        middle_lvl = self.step[s] * n_lvl
        for i in range(1,self.amount_lvl // 2+1):
            lvl = middle_lvl + self.step[s]*i
            self.lvls[s].append(lvl)
            lvl = middle_lvl - self.step[s]*i
            self.lvls[s].append(lvl)
        self.lvls[s].append(middle_lvl)
        self.lvls[s].sort()
        self.up_lvls[s] = max(self.lvls[s]) + self.step[s] * self.reset_n
        self.down_lvls[s] = min(self.lvls[s]) - self.step[s] * self.reset_n
        self.middle_lvls[s] = middle_lvl

    def long_grid(self,row,s):
        new_pos = -1
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if row['close'] <= lvl:
                new_pos += 1
            elif lvl + buff >= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        elif new_pos == -1:
            if self.keep:
                new_pos = None
            else:
                new_pos = 0
        self.need_pos[s] = new_pos

    
    def short_grid(self,row,s):
        new_pos = 1
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if row['close'] >= lvl:
                new_pos -= 1
            elif lvl - buff <= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        elif new_pos == 1:
            if self.keep:
                new_pos = None
            else:
                new_pos = 0
        self.need_pos[s] = new_pos
    
    def neutral_grid(self,row,s):
        new_pos = 0
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if lvl < self.middle_lvls[s]:
                if row['close'] <= lvl:
                    new_pos += 1
                elif lvl + buff >= row['close'] and self.positions[s] > 0:
                    new_pos = None
                    break
            elif lvl > self.middle_lvls[s]:
                if row['close'] >= lvl:
                    new_pos -= 1
                elif lvl - buff <= row['close'] and self.positions[s] < 0:
                    new_pos = None
                    break
        if new_pos == 0:
            new_pos = None
        self.need_pos[s] = new_pos
        
    
    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]
            if self.first_run[s]:
                self.first_run[s] = False
                self.init_grid(s,row)
                print('LWS2_SWIMGRID:',s,self.lvls[s],self.up_lvls[s],self.down_lvls[s])
            self.update_grid(s,row)
            self.grid_func(row,s)
            # print(self.lvls[s],row['close'],self.need_pos[s])
        return self.need_pos

class LWS2_PSG(WSBase):
    """парный реверс плавающий грид-бот c автоматическим рассчетом уровней"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'amount_lvl': 5,
            'per_step':0.1,
            'keep':False,
            'reset_n':2
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.amount_lvl = parameters['amount_lvl']
        self.first_run = {s: True for s in self.symbols}
        self.step = {s: None for s in self.symbols}
        self.lvls = {s: list() for s in self.symbols}
        self.per_step = parameters['per_step']
        self.buff = self.per_step / 2
        self.keep = parameters['keep']
        self.grid_dirs = {s: 1 for s in self.symbols}
        self.grid_dirs[self.symbols[1]] = -1
        self.up_lvls = {s: None for s in self.symbols}
        self.down_lvls = {s: None for s in self.symbols}
        self.reset_n = parameters['reset_n']

    def update_grid(self,s,row):
        self.step[s] = (row['close'] / 100) * self.per_step
        if self.grid_dirs[s] == 1: #long
            self.long_init_grid(s,row)
        else:
            self.short_init_grid(s,row)

    def grid_func(self,row,s):
        if self.grid_dirs[s] == 1: #long
            self.long_grid(row,s)
        else:
            self.short_grid(row,s)

    def long_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        start_lvl = self.step[s] * n_lvl
        for i in range(self.amount_lvl):
            lvl = start_lvl - self.step[s]*i
            self.lvls[s].append(lvl)
        self.up_lvls[s] = start_lvl + self.step[s] * self.reset_n
        self.down_lvls[s] = start_lvl - self.step[s] * (self.amount_lvl + self.reset_n - 1)

    def short_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        start_lvl = self.step[s] * n_lvl
        for i in range(self.amount_lvl):
            lvl = start_lvl + self.step[s]*i
            self.lvls[s].append(lvl)
        self.up_lvls[s] = start_lvl + self.step[s] * (self.amount_lvl + self.reset_n - 1)
        self.down_lvls[s] = start_lvl - self.step[s] * self.reset_n

    def long_grid(self,row,s):
        new_pos = -1
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if row['close'] <= lvl:
                new_pos += 1
            elif lvl + buff >= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        elif new_pos == -1:
            if self.keep:
                new_pos = None
            else:
                new_pos = 0
        self.need_pos[s] = new_pos

    
    def short_grid(self,row,s):
        new_pos = 1
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if row['close'] >= lvl:
                new_pos -= 1
            elif lvl - buff <= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        elif new_pos == 1:
            if self.keep:
                new_pos = None
            else:
                new_pos = 0
        self.need_pos[s] = new_pos
        
    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]
            if self.first_run[s]:
                self.first_run[s] = False
                self.update_grid(s,row)
                print('LWS2_PSG:',s,self.grid_dirs[s],self.lvls[s],self.up_lvls[s],self.down_lvls[s])
            if row['close'] > self.up_lvls[s] or row['close'] < self.down_lvls[s]:
                self.update_grid(s,row)
                # print(s,row['close'],self.lvls[s])
            self.grid_func(row,s)
            # print(s,row['close'],self.lvls[s],self.need_pos[s])
        return self.need_pos
    
class LWS2_PSGSON(WSBase):
    """ИСП1 парный реверс плавающий грид-бот c автоматическим рассчетом уровней"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'amount_lvl': 5,
            'per_step':0.1,
            'keep':False,
            'reset_n':2
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.amount_lvl = parameters['amount_lvl']
        self.first_run = {s: True for s in self.symbols}
        self.step = {s: None for s in self.symbols}
        self.lvls = {s: list() for s in self.symbols}
        self.per_step = parameters['per_step']
        self.buff = self.per_step / 2
        self.keep = parameters['keep']
        self.grid_dirs = {s: 1 for s in self.symbols}
        self.grid_dirs[self.symbols[1]] = -1
        self.up_lvls = {s: None for s in self.symbols}
        self.down_lvls = {s: None for s in self.symbols}
        self.reset_n = parameters['reset_n']

    def init_grid(self,s,row):
        self.step[s] = (row['close'] / 100) * self.per_step
        if self.grid_dirs[s] == 1: #long
            self.long_init_grid(s,row)
        else:
            self.short_init_grid(s,row)

    def update_grid(self,s,row):
        if row['close'] > self.up_lvls[s]:
            self.step[s] = (row['close'] / 100) * self.per_step
            self.long_init_grid(s,row)
        elif row['close'] < self.down_lvls[s]:
            self.step[s] = (row['close'] / 100) * self.per_step
            self.short_init_grid(s,row)


    def grid_func(self,row,s):
        if self.grid_dirs[s] == 1: #long
            self.long_grid(row,s)
        else:
            self.short_grid(row,s)

    def long_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        start_lvl = self.step[s] * n_lvl
        for i in range(self.amount_lvl):
            lvl = start_lvl - self.step[s]*i
            self.lvls[s].append(lvl)
        self.up_lvls[s] = start_lvl + self.step[s] * self.reset_n
        self.down_lvls[s] = start_lvl - self.step[s] * (self.amount_lvl + self.reset_n - 1)

    def short_init_grid(self,s,row):
        self.lvls[s].clear()
        n_lvl = row['close'] // self.step[s]
        start_lvl = self.step[s] * n_lvl
        for i in range(self.amount_lvl):
            lvl = start_lvl + self.step[s]*i
            self.lvls[s].append(lvl)
        self.up_lvls[s] = start_lvl + self.step[s] * (self.amount_lvl + self.reset_n - 1)
        self.down_lvls[s] = start_lvl - self.step[s] * self.reset_n

    def long_grid(self,row,s):
        new_pos = -1
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if row['close'] <= lvl:
                new_pos += 1
            elif lvl + buff >= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        elif new_pos == -1:
            if self.keep:
                new_pos = None
            else:
                new_pos = 0
        self.need_pos[s] = new_pos

    
    def short_grid(self,row,s):
        new_pos = 1
        buff = self.step[s] / 2
        for lvl in self.lvls[s]:
            if row['close'] >= lvl:
                new_pos -= 1
            elif lvl - buff <= row['close']:
                new_pos = None
                break
        if new_pos == 0:
            new_pos = None
        elif new_pos == 1:
            if self.keep:
                new_pos = None
            else:
                new_pos = 0
        self.need_pos[s] = new_pos
        
    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]
            if self.first_run[s]:
                self.first_run[s] = False
                self.init_grid(s,row)
                print('LWS2_PSG:',s,self.grid_dirs[s],self.lvls[s],self.up_lvls[s],self.down_lvls[s])
            self.update_grid(s,row)
                # print(s,row['close'],self.lvls[s])
            self.grid_func(row,s)
            # print(s,row['close'],self.lvls[s],self.need_pos[s])
        return self.need_pos
    
class LWS3_NEXUS(WSBase):
    """Парный реверс хедж"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'period_rsi':14,
            'lvls':(10,30,70,90),
            'buff':10,
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.period_rsi = parameters['period_rsi']
        self.lvls = sorted(parameters['lvls'])
        self.buff = parameters['buff']
        self.grid_dirs = {s: 1 for s in self.symbols}
        self.grid_dirs[self.symbols[1]] = -1
        
    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        for tf in self.last_dfs:
            for s in self.last_dfs[tf]:
                df = self.last_dfs[tf][s]
                df = add_rsi(df,self.period_rsi)
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]
            rsi = row['rsi']
            
            if self.grid_dirs[s] == 1:  # LONG
                if rsi <= self.lvls[0]: # 0-10
                    npos = 3
                elif self.lvls[1] - self.buff <= rsi <= self.lvls[1]: #20-30
                    npos = 2
                elif self.lvls[2] - self.buff <= rsi <= self.lvls[2]: #60-70
                    npos = 1
                elif rsi >= self.lvls[3]: #90-100
                    npos = 0
                else:
                    npos = None
                    
            else:  # SHORT
                if rsi <= self.lvls[0]: # 0-10
                    npos = 0
                elif self.lvls[1] + self.buff >= rsi >= self.lvls[1]: #30-40
                    npos = -1
                elif self.lvls[2] + self.buff >= rsi >= self.lvls[2]: #70-80
                    npos = -2
                elif rsi >= self.lvls[3]: #90-100
                    npos = -3
                else:
                    npos = None

            self.need_pos[s] = npos
            # print(s,row['rsi'],npos)
        return self.need_pos
    
class LWS3_APEX(WSBase):
    """Парный реверс хедж"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'period_rsi':14,
            'lvls':(10,30,70,90),
            'buff':10,
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.period_rsi = parameters['period_rsi']
        self.lvls = sorted(parameters['lvls'])
        self.buff = parameters['buff']
        self.grid_dirs = {s: 1 for s in self.symbols}
        self.grid_dirs[self.symbols[1]] = -1
        
    def preprocessing(self, dfs, poss):
        self.last_dfs = dfs.copy()
        for tf in self.last_dfs:
            for s in self.last_dfs[tf]:
                df = self.last_dfs[tf][s]
                df = add_rsi(df,self.period_rsi)
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]
            rsi = row['rsi']
            
            if self.grid_dirs[s] == 1:  # LONG
                if rsi <= self.lvls[0]: # 0-10
                    npos = 4
                elif self.lvls[1] - self.buff <= rsi <= self.lvls[1]: #20-30
                    npos = 3
                elif 50 <= rsi <= 50 + self.buff: #50-60
                    npos = 2
                elif self.lvls[2] + self.buff >= rsi >= self.lvls[2]: #70-80
                    npos = 1
                elif rsi >= self.lvls[3]: #90-100
                    npos = 0
                else:
                    npos = None
                    
            else:  # SHORT
                if rsi <= self.lvls[0]: # 0-10
                    npos = 0
                elif self.lvls[1] - self.buff <= rsi <= self.lvls[1]: #20-30
                    npos = -1
                elif 50 >= rsi >= 50 - self.buff: #40-50
                    npos = -2
                elif self.lvls[2] + self.buff >= rsi >= self.lvls[2]: #70-80
                    npos = -3
                elif rsi >= self.lvls[3]: #90-100
                    npos = -4
                else:
                    npos = None

            self.need_pos[s] = npos
            # print(s,row['rsi'],npos)
        return self.need_pos


class LWS4_SWATR(WSBase):
    """Плавающий грид-бот c ATR"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'amount_lvl': 5,
            'atr_multiplier': 2.0,
            'atr_period': 14,
            'grid_dir': 1,
            'keep': False,
            'reset_n': 2,
            'smoothing_factor': 0.1,
            'buffer_multiplier': 0.3  
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.amount_lvl = parameters['amount_lvl']
        self.first_run = {s: True for s in self.symbols}
        self.step = {s: None for s in self.symbols}
        self.lvls = {s: list() for s in self.symbols}
        self.atr_multiplier = parameters['atr_multiplier']
        self.atr_period = parameters.get('atr_period', 14)
        self.smoothing_factor = parameters.get('smoothing_factor', 0.1)
        self.buffer_multiplier = parameters.get('buffer_multiplier', 0.3)  # Новый параметр
        self.grid_dir = parameters['grid_dir']
        self.keep = parameters['keep']
        self.reset_n = parameters['reset_n']
        self.up_lvls = {s: None for s in self.symbols}
        self.down_lvls = {s: None for s in self.symbols}
        self.middle_lvls = {s: None for s in self.symbols}
        self.atr_values = {s: None for s in self.symbols}
        self.smoothed_atr = {s: None for s in self.symbols}
        
        if self.grid_dir == 1:
            self.grid_func = self.long_grid
        elif self.grid_dir == -1:
            self.grid_func = self.short_grid
        else:
            self.grid_func = self.neutral_grid
    
    def calculate_atr(self, s, row, dfs):
        """Расчет ATR на основе исторических данных"""
        tf1 = self.timeframes[0]
        df = dfs[tf1][s]
        
        # Добавляем ATR к данным если его нет
        if 'atr' not in df.columns:
            df = add_atr(df, self.atr_period)
            dfs[tf1][s] = df
        
        current_atr = df['atr'].iloc[-1]
        
        # Сглаживание ATR для избежания резких изменений
        if self.smoothed_atr[s] is None:
            self.smoothed_atr[s] = current_atr
        else:
            self.smoothed_atr[s] = (self.smoothing_factor * current_atr + 
                                   (1 - self.smoothing_factor) * self.smoothed_atr[s])
        
        self.atr_values[s] = self.smoothed_atr[s]
        self.step[s] = self.atr_values[s] * self.atr_multiplier
        return self.step[s]

    def init_grid(self, s, row, dfs):
        """Инициализация сетки с ATR"""
        step = self.calculate_atr(s, row, dfs)
        
        # Минимальный шаг для избежания слишком частых переключений
        min_step = row['close'] * 0.002
        self.step[s] = max(step, min_step)
        
        if self.grid_dir == 1:
            self.long_init_grid(s, row)
        elif self.grid_dir == -1:
            self.short_init_grid(s, row)
        else:
            self.neutral_init_grid(s, row)

    def update_grid(self, s, row, dfs):
        """Обновление сетки с проверкой гистерезиса"""
        step = self.calculate_atr(s, row, dfs)
        
        # Гистерезис: обновляем сетку только если цена вышла за границы + буфер ATR
        atr_buffer = self.atr_values[s] * 1.5  # Буфер в 1.5 ATR
        
        up_threshold = self.up_lvls[s] + atr_buffer if self.grid_dir != -1 else self.up_lvls[s]
        down_threshold = self.down_lvls[s] - atr_buffer if self.grid_dir != 1 else self.down_lvls[s]
        
        if row['close'] > up_threshold:
            self.step[s] = step
            if self.grid_dir != 0:
                self.long_init_grid(s, row)
            else:
                self.neutral_init_grid(s, row)
            # print(f'LWS4_SWATR: {s} - Grid updated UP. New levels: {self.lvls[s]}')
            
        elif row['close'] < down_threshold:
            self.step[s] = step
            if self.grid_dir != 0:
                self.short_init_grid(s, row)
            else:
                self.neutral_init_grid(s, row)
            # print(f'LWS4_SWATR: {s} - Grid updated DOWN. New levels: {self.lvls[s]}')

    def long_init_grid(self, s, row):
        """Инициализация лонг сетки с округлением до шага ATR"""
        self.lvls[s].clear()
        # Округление цены до ближайшего шага ATR
        n_lvl = round(row['close'] / self.step[s])
        start_lvl = self.step[s] * n_lvl
        
        for i in range(self.amount_lvl):
            lvl = start_lvl - self.step[s] * i
            self.lvls[s].append(round(lvl, 6))  # Округление для точности
            
        self.up_lvls[s] = start_lvl + self.step[s] * self.reset_n
        self.down_lvls[s] = start_lvl - self.step[s] * (self.amount_lvl + self.reset_n - 1)

    def short_init_grid(self, s, row):
        """Инициализация шорт сетки с округлением до шага ATR"""
        self.lvls[s].clear()
        n_lvl = round(row['close'] / self.step[s])
        start_lvl = self.step[s] * n_lvl
        
        for i in range(self.amount_lvl):
            lvl = start_lvl + self.step[s] * i
            self.lvls[s].append(round(lvl, 6))
            
        self.up_lvls[s] = start_lvl + self.step[s] * (self.amount_lvl + self.reset_n - 1)
        self.down_lvls[s] = start_lvl - self.step[s] * self.reset_n

    def neutral_init_grid(self, s, row):
        """Инициализация нейтральной сетки с округлением до шага ATR"""
        self.lvls[s].clear()
        n_lvl = round(row['close'] / self.step[s])
        middle_lvl = self.step[s] * n_lvl
        
        # Создаем уровни в обе стороны от средней цены
        for i in range(1, self.amount_lvl // 2 + 1):
            lvl = middle_lvl + self.step[s] * i
            self.lvls[s].append(round(lvl, 6))
            lvl = middle_lvl - self.step[s] * i
            self.lvls[s].append(round(lvl, 6))
            
        self.lvls[s].append(round(middle_lvl, 6))
        self.lvls[s].sort()
        self.up_lvls[s] = max(self.lvls[s]) + self.step[s] * self.reset_n
        self.down_lvls[s] = min(self.lvls[s]) - self.step[s] * self.reset_n
        self.middle_lvls[s] = middle_lvl

    def long_grid(self, row, s):
        """Лонг сетка с динамическим буфером на основе ATR"""
        new_pos = -1
        dynamic_buffer = self.atr_values[s] * self.buffer_multiplier
        
        for lvl in self.lvls[s]:
            if row['close'] <= lvl:
                new_pos += 1
            elif lvl + dynamic_buffer >= row['close']:
                new_pos = None
                break
                
        if new_pos == 0:
            new_pos = None
        elif new_pos == -1:
            new_pos = 0 if not self.keep else None
            
        self.need_pos[s] = new_pos

    def short_grid(self, row, s):
        """Шорт сетка с динамическим буфером"""
        new_pos = 1
        dynamic_buffer = self.atr_values[s] * self.buffer_multiplier
        
        for lvl in self.lvls[s]:
            if row['close'] >= lvl:
                new_pos -= 1
            elif lvl - dynamic_buffer <= row['close']:
                new_pos = None
                break
                
        if new_pos == 0:
            new_pos = None
        elif new_pos == 1:
            new_pos = 0 if not self.keep else None
            
        self.need_pos[s] = new_pos
    
    def neutral_grid(self, row, s):
        """Нейтральная сетка с динамическим буфером ATR"""
        new_pos = 0
        dynamic_buffer = self.atr_values[s] * self.buffer_multiplier
        
        for lvl in self.lvls[s]:
            if lvl < self.middle_lvls[s]:
                # Уровни ниже средней цены - для покупок
                if row['close'] <= lvl:
                    new_pos += 1
                elif lvl + dynamic_buffer >= row['close'] and self.positions[s] > 0:
                    new_pos = None
                    break
            elif lvl > self.middle_lvls[s]:
                # Уровни выше средней цены - для продаж
                if row['close'] >= lvl:
                    new_pos -= 1
                elif lvl - dynamic_buffer <= row['close'] and self.positions[s] < 0:
                    new_pos = None
                    break
                    
        if new_pos == 0:
            new_pos = None
            
        self.need_pos[s] = new_pos
    
    def preprocessing(self, dfs, poss):
        """Предобработка с расчетом ATR для всех символов"""
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        
        for s in self.symbols:
            if s in dfs[tf1]:
                # Гарантируем, что ATR рассчитан для всех данных
                df = dfs[tf1][s].copy()
                if 'atr' not in df.columns:
                    df = add_atr(df, self.atr_period)
                self.last_dfs[tf1][s] = df
                
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        """Основной метод с интеграцией ATR"""
        tf1 = self.timeframes[0]
        for s in self.symbols:
            if s in self.last_dfs[tf1]:
                df = self.last_dfs[tf1][s]
                row = df.iloc[-1]
                
                if self.first_run[s]:
                    self.first_run[s] = False
                    self.init_grid(s, row, self.last_dfs)
                    print(f'LWS4_SWATR: {s}, Levels: {[round(l, 6) for l in self.lvls[s]]}, '
                          f'Step: {self.step[s]:.6f}, ATR: {self.atr_values[s]:.6f}')
                
                self.update_grid(s, row, self.last_dfs)
                self.grid_func(row, s)
                
        return self.need_pos
