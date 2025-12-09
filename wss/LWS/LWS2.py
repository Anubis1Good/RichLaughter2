import pandas as pd
from wss.WSBase import WSBase
from indicators.classic_ind import add_rsi,add_atr

class LWS5_ARCHIMEDES(WSBase):
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
        self.lvls = list(parameters['lvls'])
        self.us_lvl = parameters['us_lvl']
        self.ds_lvl = parameters['ds_lvl']
        self.grid_dir = parameters['grid_dir']
        self.middle_lvl = sum(self.lvls) / len(self.lvls)
        self.in_work = True
        if self.grid_dir == 1: #long
            self.grid_func = self.long_grid
            self.lvls.sort(reverse=True)
        elif self.grid_dir == -1:
            self.grid_func = self.short_grid
            self.lvls.sort()
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
        new_pos = None
        max_pos = 1
        for i,lvl in enumerate(self.lvls):
            if row['close'] <= lvl and row['high_1'] > lvl:
                new_pos = i + 1
            if row['close'] <= lvl:
                max_pos += 1
        if self.positions[s] > max_pos:
            new_pos = max_pos
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
        new_pos = None
        max_pos = -1
        for i,lvl in enumerate(self.lvls):
            if row['close'] >= lvl and row['low_1'] < lvl:
                new_pos = -(i + 1)
            if row['close'] >= lvl:
                max_pos -= 1
        if self.positions[s] < max_pos:
            new_pos = max_pos
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
        new_pos = None
        max_pos = 0
        for i,lvl in enumerate(self.lvls):
            if lvl < self.middle_lvl:
                if row['close'] <= lvl:
                    max_pos += 1
                if row['close'] <= lvl and row['high_1'] > lvl:
                    new_pos = 1
            elif lvl > self.middle_lvl:
                if row['close'] >= lvl:
                    max_pos -= 1
                if row['close'] >= lvl and row['low_1'] < lvl:
                    new_pos = -1
        if new_pos:
            new_pos = max_pos
        if self.positions[s] > 0 and max_pos < 0:
            new_pos = max_pos
        elif self.positions[s] < 0 and max_pos > 0:
            new_pos = max_pos
        self.need_pos[s] = new_pos
        return True
    
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            df['high_1'] = df['high'].shift(1)
            df['low_1'] = df['low'].shift(1)
            self.last_dfs[tf1][s] = df
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
    
class LWS5_CADUCEUS(WSBase):
    """грид-бот c авто рассчетом"""
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
        delta_se = parameters['end'] - parameters['start']
        step_lvl = delta_se / (parameters['amount_lvl'] - 1)
        self.lvls = [parameters['start'] + step_lvl*i for i in range(parameters['amount_lvl'])]
        print(symbols,self.lvls)
        self.us_lvl = parameters['us_lvl']
        self.ds_lvl = parameters['ds_lvl']
        self.grid_dir = parameters['grid_dir']
        self.middle_lvl = sum(self.lvls) / len(self.lvls)
        self.in_work = True
        if self.grid_dir == 1: #long
            self.grid_func = self.long_grid
            self.lvls.sort(reverse=True)
        elif self.grid_dir == -1:
            self.grid_func = self.short_grid
            self.lvls.sort()
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
        new_pos = None
        max_pos = 1
        for i,lvl in enumerate(self.lvls):
            if row['close'] <= lvl and row['high_1'] > lvl:
                new_pos = i + 1
            if row['close'] <= lvl:
                max_pos += 1
        if self.positions[s] > max_pos:
            new_pos = max_pos
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
        new_pos = None
        max_pos = -1
        for i,lvl in enumerate(self.lvls):
            if row['close'] >= lvl and row['low_1'] < lvl:
                new_pos = -(i + 1)
            if row['close'] >= lvl:
                max_pos -= 1
        if self.positions[s] < max_pos:
            new_pos = max_pos
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
        new_pos = None
        max_pos = 0
        for i,lvl in enumerate(self.lvls):
            if lvl < self.middle_lvl:
                if row['close'] <= lvl:
                    max_pos += 1
                if row['close'] <= lvl and row['high_1'] > lvl:
                    new_pos = 1
            elif lvl > self.middle_lvl:
                if row['close'] >= lvl:
                    max_pos -= 1
                if row['close'] >= lvl and row['low_1'] < lvl:
                    new_pos = -1
        if new_pos:
            new_pos = max_pos
        if self.positions[s] > 0 and max_pos < 0:
            new_pos = max_pos
        elif self.positions[s] < 0 and max_pos > 0:
            new_pos = max_pos
        self.need_pos[s] = new_pos
        return True
    
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            df['high_1'] = df['high'].shift(1)
            df['low_1'] = df['low'].shift(1)
            self.last_dfs[tf1][s] = df
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
    
class LWS5_PROGRESSO(WSBase):
    """x-усредняющий грид-бот c авто рассчетом"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'start':200,
            'amount_lvl': 5,
            'start_step':0.5,
            'mult_lvl': 2,
            'us_lvl': None,
            'ds_lvl': None,
            'grid_dir': 1,
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        
        self.us_lvl = parameters['us_lvl']
        self.ds_lvl = parameters['ds_lvl']
        self.grid_dir = parameters['grid_dir']
        self.in_work = True
        step_per = (parameters['start_step'] * parameters['start']) / 100
        if self.grid_dir == 1:  # long
            # Уровни для лонга: ниже стартовой цены
            self.lvls = [parameters['start'] - step_per * (parameters['mult_lvl']**i - 1) 
                        for i in range(parameters['amount_lvl'])]
            self.grid_func = self.long_grid
            self.lvls.sort(reverse=True)
            
        elif self.grid_dir == -1:  # short
            # Уровни для шорта: выше стартовой цены
            self.lvls = [parameters['start'] + step_per * (parameters['mult_lvl']**i - 1) 
                        for i in range(parameters['amount_lvl'])]
            self.grid_func = self.short_grid
            self.lvls.sort()
            
        else:  # neutral
            total_lvls = parameters['amount_lvl']
            # Для нечетного количества: больше уровней в одной стороне
            lower_count = (total_lvls + 1) // 2  # округляем вверх
            upper_count = total_lvls // 2        # округляем вниз
            
            # Уровни ниже или на стартовой цене
            lower_lvls = [parameters['start'] - step_per * (parameters['mult_lvl']**i - 1) 
                        for i in range(lower_count)]
            
            # Уровни выше стартовой цены (начинаем с i=1)
            upper_lvls = [parameters['start'] + step_per * (parameters['mult_lvl']**i - 1) 
                        for i in range(1, upper_count + 1)]
            
            self.lvls = lower_lvls + upper_lvls
            self.grid_func = self.neutral_grid
            self.lvls.sort()
        
        self.middle_lvl = sum(self.lvls) / len(self.lvls)
        print(symbols, self.lvls)
    
    def long_grid(self,row,s):
        if self.ds_lvl:
            if row['close'] < self.ds_lvl:
                self.in_work = False
                return False    
        if self.us_lvl:
            if row['close'] > self.us_lvl:
                self.need_pos[s] = 0
                return True
        new_pos = None
        max_pos = 1
        for i,lvl in enumerate(self.lvls):
            if row['close'] <= lvl and row['high_1'] > lvl:
                new_pos = i + 1
            if row['close'] <= lvl:
                max_pos += 1
        if self.positions[s] > max_pos:
            new_pos = max_pos
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
        new_pos = None
        max_pos = -1
        for i,lvl in enumerate(self.lvls):
            if row['close'] >= lvl and row['low_1'] < lvl:
                new_pos = -(i + 1)
            if row['close'] >= lvl:
                max_pos -= 1
        if self.positions[s] < max_pos:
            new_pos = max_pos
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
        new_pos = None
        max_pos = 0
        for i,lvl in enumerate(self.lvls):
            if lvl < self.middle_lvl:
                if row['close'] <= lvl:
                    max_pos += 1
                if row['close'] <= lvl and row['high_1'] > lvl:
                    new_pos = 1
            elif lvl > self.middle_lvl:
                if row['close'] >= lvl:
                    max_pos -= 1
                if row['close'] >= lvl and row['low_1'] < lvl:
                    new_pos = -1
        if new_pos:
            new_pos = max_pos
        if self.positions[s] > 0 and max_pos < 0:
            new_pos = max_pos
        elif self.positions[s] < 0 and max_pos > 0:
            new_pos = max_pos
        self.need_pos[s] = new_pos
        return True
    
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            df['high_1'] = df['high'].shift(1)
            df['low_1'] = df['low'].shift(1)
            self.last_dfs[tf1][s] = df
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
    
class LWS5_XPG(WSBase):
    """ИСП1 x-усредняющий грид-бот c авто рассчетом"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'start':200,
            'amount_lvl': 5,
            'start_step':0.5,
            'mult_lvl': 2,
            'us_lvl': None,
            'ds_lvl': None,
            'grid_dir': 1,
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        
        self.us_lvl = parameters['us_lvl']
        self.ds_lvl = parameters['ds_lvl']
        self.grid_dir = parameters['grid_dir']
        self.in_work = True
        step_per = (parameters['start_step'] * parameters['start']) / 100
        if self.grid_dir == 1:  # long
            # Уровни для лонга: ниже стартовой цены
            self.lvls = [parameters['start'] - step_per * (parameters['mult_lvl']**i - 1) 
                        for i in range(parameters['amount_lvl'])]
            self.grid_func = self.long_grid
            self.lvls.sort(reverse=True)
            
        elif self.grid_dir == -1:  # short
            # Уровни для шорта: выше стартовой цены
            self.lvls = [parameters['start'] + step_per * (parameters['mult_lvl']**i - 1) 
                        for i in range(parameters['amount_lvl'])]
            self.grid_func = self.short_grid
            self.lvls.sort()
            
        else:  # neutral
            total_lvls = parameters['amount_lvl']
            # Уровни ниже или на стартовой цене
            lower_lvls = [parameters['start'] - step_per * (parameters['mult_lvl']**i - 1) 
                        for i in range(total_lvls+1)]
            
            # Уровни выше стартовой цены (начинаем с i=1)
            upper_lvls = [parameters['start'] + step_per * (parameters['mult_lvl']**i - 1) 
                        for i in range(1, total_lvls + 1)]
            
            self.lvls = lower_lvls + upper_lvls
            self.grid_func = self.neutral_grid
            self.lvls.sort()
        
        self.middle_lvl = sum(self.lvls) / len(self.lvls)
        print(symbols, self.lvls)
    
    def long_grid(self,row,s):
        if self.ds_lvl:
            if row['close'] < self.ds_lvl:
                self.in_work = False
                return False    
        if self.us_lvl:
            if row['close'] > self.us_lvl:
                self.need_pos[s] = 0
                return True
        new_pos = None
        max_pos = 1
        for lvl in self.lvls:
            if row['close'] <= lvl:
                max_pos += 1
                new_pos = max_pos - 1
        if self.positions[s] >= max_pos:
            new_pos = max_pos
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
        new_pos = None
        max_pos = -1
        for lvl in self.lvls:
            if row['close'] >= lvl:
                max_pos -= 1
                new_pos = max_pos + 1
        if self.positions[s] <= max_pos:
            new_pos = max_pos
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
        new_pos = None
        long = row['close'] < self.middle_lvl
        max_pos = 1 if long else -1
        for lvl in self.lvls:
            if long:
                if lvl < self.middle_lvl:
                    if row['close'] <= lvl:
                        max_pos += 1
                        new_pos = max_pos - 1
                else:
                    continue
            else:
                if lvl > self.middle_lvl:
                    if row['close'] >= lvl:
                        max_pos -= 1
                        new_pos = max_pos + 1
                else:
                    continue
        if long:
            if self.positions[s] >= max_pos:
                new_pos = max_pos
        else:
            if self.positions[s] <= max_pos:
                new_pos = max_pos  
        self.need_pos[s] = new_pos
        return True
    
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
            for s in self.last_dfs[tf1]:
                row = self.last_dfs[tf1][s].iloc[-1]
                if not self.grid_func(row,s):
                    break
        else:
            self.need_pos = {s: 0 for s in self.symbols}
        return self.need_pos
    
class LWS6_TIDAL(WSBase):
    """грид-бот c авто рассчетом и выходом по rsi"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'start':200,
            'end':300,
            'amount_lvl': 5,
            'period_rsi':14,
            'rsi_thresh':30,
            'us_lvl': None,
            'ds_lvl': 100,
            'grid_dir': 1,
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        delta_se = parameters['end'] - parameters['start']
        step_lvl = delta_se / (parameters['amount_lvl'] - 1)
        self.lvls = [parameters['start'] + step_lvl*i for i in range(parameters['amount_lvl'])]
        print(symbols,self.lvls)
        self.us_lvl = parameters['us_lvl']
        self.ds_lvl = parameters['ds_lvl']
        self.grid_dir = parameters['grid_dir']
        self.middle_lvl = sum(self.lvls) / len(self.lvls)
        self.in_work = True
        self.period_rsi = parameters['period_rsi']
        self.rsi_thresh = parameters['rsi_thresh']
        if self.grid_dir == 1: #long
            self.grid_func = self.long_grid
            self.lvls.sort(reverse=True)
        elif self.grid_dir == -1:
            self.grid_func = self.short_grid
            self.lvls.sort()
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
        new_pos = None
        max_pos = 1
        for i,lvl in enumerate(self.lvls):
            if row['rsi'] < 70 - self.rsi_thresh:
                if row['close'] <= lvl and row['high_1'] > lvl:
                    new_pos = i + 1
            if row['close'] <= lvl:
                max_pos += 1
        if row['rsi'] > 100 - self.rsi_thresh:
            new_pos = max_pos
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
        new_pos = None
        max_pos = -1
        for i,lvl in enumerate(self.lvls):
            if row['rsi'] > 30 + self.rsi_thresh:
                if row['close'] >= lvl and row['low_1'] < lvl:
                    new_pos = -(i + 1)
            if row['close'] >= lvl:
                max_pos -= 1
        if row['rsi'] < self.rsi_thresh:
            new_pos = max_pos
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
        new_pos = None
        max_pos = 0
        for i,lvl in enumerate(self.lvls):
            if lvl < self.middle_lvl:
                if row['close'] <= lvl:
                    max_pos += 1
                if row['close'] <= lvl and row['high_1'] > lvl:
                    new_pos = 1
            elif lvl > self.middle_lvl:
                if row['close'] >= lvl:
                    max_pos -= 1
                if row['close'] >= lvl and row['low_1'] < lvl:
                    new_pos = -1
        if new_pos:
            new_pos = max_pos
        if row['rsi'] > 100 - self.rsi_thresh and max_pos < 0:
            new_pos = max_pos
        elif row['rsi'] < self.rsi_thresh and max_pos > 0:
            new_pos = max_pos
        self.need_pos[s] = new_pos
        return True
    
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            df = add_rsi(df,self.period_rsi)
            df['high_1'] = df['high'].shift(1)
            df['low_1'] = df['low'].shift(1)
            self.last_dfs[tf1][s] = df
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
    

class LWS7_FLOATGRESSO(WSBase):
    """Х-усредняющий плавающий грид-бот c авто уровнями"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'amount_lvl': 3,
            'offset':0.1,
            'mult_lvl': 2,
            'grid_dir': 1,
            'reset_n':2,
            'pariod_grid':150,
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.amount_lvl = parameters['amount_lvl']
        self.mult_lvl = parameters['mult_lvl']
        self.period_grid = parameters['pariod_grid']
        self.first_run = {s: True for s in self.symbols}
        self.step = {s: None for s in self.symbols}
        self.lvls = {s: list() for s in self.symbols}
        self.offset = parameters['offset']
        self.grid_dir = parameters['grid_dir']
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
        self.step[s] = row['average_period'] * self.offset / 100
        if self.grid_dir == 1: #long
            self.long_init_grid(s,row)
        elif self.grid_dir == -1:
            self.short_init_grid(s,row)
        else:
            self.neutral_init_grid(s,row)

    def long_init_grid(self,s,row):
        self.lvls[s].clear()
        step = self.step[s]
        first_lvl = row['max_period'] - step
        self.up_lvls[s] = row['max_period'] + step*self.reset_n
        self.lvls[s].append(first_lvl)
        
        current_step = step * self.mult_lvl  # первый множитель
        current_lvl = first_lvl
        
        for i in range(1, self.amount_lvl):
            current_lvl = current_lvl - current_step
            self.lvls[s].append(current_lvl)
            current_step *= self.mult_lvl  # увеличиваем шаг в mult_lvl раз
        
        self.lvls[s].sort(reverse=True)
        self.down_lvls[s] = min(self.lvls[s]) - step*self.reset_n
        self.middle_lvls[s] = row['average_period']
    

    def short_init_grid(self,s,row):
        self.lvls[s].clear()
        step = self.step[s]
        first_lvl = row['min_period'] + step
        self.down_lvls[s] = row['min_period'] - step*self.reset_n
        self.lvls[s].append(first_lvl)
        
        current_step = step * self.mult_lvl  # первый множитель
        current_lvl = first_lvl
        
        for i in range(1, self.amount_lvl):
            current_lvl = current_lvl + current_step
            self.lvls[s].append(current_lvl)
            current_step *= self.mult_lvl  # увеличиваем шаг в mult_lvl раз
        
        self.lvls[s].sort()
        self.up_lvls[s] = max(self.lvls[s]) + step*self.reset_n
        self.middle_lvls[s] = row['average_period']

    def neutral_init_grid(self,s,row):
        self.lvls[s].clear()
        step = self.step[s]
        first_up_lvl = row['average_period'] + step
        first_down_lvl = row['average_period'] - step
        self.lvls[s].append(first_up_lvl)
        self.lvls[s].append(first_down_lvl)
        current_step = step * self.mult_lvl  # первый множитель
        current_up_lvl = first_up_lvl
        current_down_lvl = first_down_lvl
        for i in range(1, self.amount_lvl):
            current_up_lvl = current_up_lvl + current_step
            self.lvls[s].append(current_up_lvl)
            current_down_lvl = current_down_lvl - current_step
            self.lvls[s].append(current_down_lvl)
            current_step *= self.mult_lvl
        self.lvls[s].sort()
        self.up_lvls[s] = max(self.lvls[s]) + step*self.reset_n
        self.down_lvls[s] = min(self.lvls[s]) - step*self.reset_n
        self.middle_lvls[s] = row['average_period']

    def long_grid(self,row,s):
        new_pos = None
        max_pos = 1
        for lvl in self.lvls[s]:
            if row['close'] <= lvl:
                max_pos += 1
                new_pos = max_pos - 1
        if self.positions[s] >= max_pos:
            new_pos = max_pos
        self.need_pos[s] = new_pos
    
    def short_grid(self,row,s):
        new_pos = None
        max_pos = -1
        for lvl in self.lvls[s]:
            if row['close'] >= lvl:
                max_pos -= 1
                new_pos = max_pos + 1
        if self.positions[s] <= max_pos:
            new_pos = max_pos
        self.need_pos[s] = new_pos
    
    def neutral_grid(self,row,s):
        new_pos = None
        long = row['close'] < self.middle_lvls[s]
        max_pos = 1 if long else -1
        for lvl in self.lvls[s]:
            if long:
                if lvl < self.middle_lvls[s]:
                    if row['close'] <= lvl:
                        max_pos += 1
                        new_pos = max_pos - 1
                else:
                    continue
            else:
                if lvl > self.middle_lvls[s]:
                    if row['close'] >= lvl:
                        max_pos -= 1
                        new_pos = max_pos + 1
                else:
                    continue
        if long:
            if self.positions[s] >= max_pos:
                new_pos = max_pos
        else:
            if self.positions[s] <= max_pos:
                new_pos = max_pos  
        self.need_pos[s] = new_pos
        
    
    def preprocessing(self, dfs, poss):
        self.last_dfs = {}
        for t in dfs:
            self.last_dfs[t] = {}
            for s in dfs[t]:
                df:pd.DataFrame = dfs[t][s].copy()
                df['max_period'] = df['high'].rolling(self.period_grid).max()
                df['min_period'] = df['low'].rolling(self.period_grid).min()
                df['average_period'] = (df['high'] + df['low']) / 2
                self.last_dfs[t][s] = df
        self.update_poss_mps(poss)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]
            if self.first_run[s]:
                self.first_run[s] = False
                self.init_grid(s,row)
                print('LWS7_FLOATGRESSO:',s,self.lvls[s],self.up_lvls[s],self.down_lvls[s], self.middle_lvls[s])
            if row['close'] > self.up_lvls[s] or row['close'] < self.down_lvls[s]:
                self.init_grid(s,row)
            self.grid_func(row,s)
            # print(self.lvls[s],row['close'],self.need_pos[s])
        return self.need_pos