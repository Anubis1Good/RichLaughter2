import pandas as pd
from wss.WSBase import WSBase
    
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
            'keep_pos':False,
            'last_point':True,
            'keep_start_long':False,
            'keep_start_short':False
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        last_point = parameters.get('last_point',False)
        self.max_pos = parameters['amount_lvl']
        self.hedge_pos = parameters['amount_lvl'] if last_point else parameters['amount_lvl'] - 1
        delta_se = parameters['end'] - parameters['start']
        step_lvl = delta_se / (parameters['amount_lvl'] - 1)
        self.lvls = [parameters['start'] + step_lvl*i for i in range(parameters['amount_lvl'])]
        print(symbols,self.lvls)
        self.uh_lvl = parameters['uh_lvl']
        self.dh_lvl = parameters['dh_lvl']
        self.first_long = parameters['first_long']
        self.keep_hedge = parameters['keep_hedge']
        self.keep_pos = parameters.get('keep_pos',False)
        self.keep_start_long = parameters.get('keep_start_long',False)
        self.keep_start_short = parameters.get('keep_start_short',False)
        self.in_work = True
        self.not_work_pos = {}
        s_l,s_s = (self.symbols[0],self.symbols[1]) if self.first_long else (self.symbols[1],self.symbols[0])
        self.not_work_pos[s_l] = self.hedge_pos
        self.not_work_pos[s_s] = -self.hedge_pos
    
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
        if self.keep_start_long:
            if new_pos_long == 0:
                self.keep_start_long = False
            else:
                if new_pos_long < cur_pos_l:
                    new_pos_long = None
        if self.keep_start_short:
            if new_pos_short == 0:
                self.keep_start_short = False
            else:
                if new_pos_short > cur_pos_s:
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
            self.need_pos = self.not_work_pos
        return self.need_pos
    
class LWS8_GRAVITON(WSBase):
    """усредняющий парный реверс грид-бот c хеджем"""
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
            'keep_pos':False,
            'last_point':True,
            'keep_start_long':False,
            'keep_start_short':False
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        last_point = parameters.get('last_point',False)
        self.max_pos = parameters['amount_lvl']
        self.hedge_pos = parameters['amount_lvl'] if last_point else parameters['amount_lvl'] - 1
        step = abs(parameters['end'] - parameters['start'])
        steps = []
        for i in range(parameters['amount_lvl']-1):
            steps.append(step)
            step /= 2

        self.long_lvls = [parameters['start'] + step for step in steps] + [parameters['start']]
        self.short_lvls = [parameters['end'] - step for step in steps] + [parameters['end']]

        print(symbols,self.long_lvls,self.short_lvls)
        self.uh_lvl = parameters['uh_lvl']
        self.dh_lvl = parameters['dh_lvl']
        self.first_long = parameters['first_long']
        self.keep_hedge = parameters['keep_hedge']
        self.keep_pos = parameters.get('keep_pos',False)
        self.keep_start_long = parameters.get('keep_start_long',False)
        self.keep_start_short = parameters.get('keep_start_short',False)
        self.in_work = True
        self.not_work_pos = {}
        s_l,s_s = (self.symbols[0],self.symbols[1]) if self.first_long else (self.symbols[1],self.symbols[0])
        self.not_work_pos[s_l] = self.hedge_pos
        self.not_work_pos[s_s] = -self.hedge_pos
    
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
        if self.keep_start_long:
            if new_pos_long == 0:
                self.keep_start_long = False
            else:
                if new_pos_long < cur_pos_l:
                    new_pos_long = None
        if self.keep_start_short:
            if new_pos_short == 0:
                self.keep_start_short = False
            else:
                if new_pos_short > cur_pos_s:
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
        for lvl in self.long_lvls:
            if row['close'] <= lvl:
                max_pos_long += 1
                new_pos_long = max_pos_long - 1
        for lvl in self.short_lvls:
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
            self.need_pos = self.not_work_pos
        return self.need_pos
    
class LWS8_LITE(WSBase):
    """парный реверс одноуровневый бот c хеджем"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'start_lvl':2500,
            'end_lvl':3000,
            'uh_lvl': 3100,
            'dh_lvl': 2400,
            'first_long': False,
            'keep_hedge':True,
            'last_point':True,
            'keep_start_long':False,
            'keep_start_short':False,
            'close_hedge':False
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        last_point = parameters.get('last_point',False)
        close_hedge = parameters.get('close_hedge',False)
        self.hedge_pos = 2 if last_point else 1
        self.hedge_pos = 0 if close_hedge else self.hedge_pos
        self.uh_lvl = parameters['uh_lvl']
        self.dh_lvl = parameters['dh_lvl']
        self.start_lvl = parameters['start_lvl']
        self.end_lvl = parameters['end_lvl']
        self.first_long = parameters['first_long']
        self.keep_hedge = parameters['keep_hedge']
        self.keep_start_long = parameters.get('keep_start_long',False)
        self.keep_start_short = parameters.get('keep_start_short',False)
        self.in_work = True
        self.not_work_pos = {}
        s_l,s_s = (self.symbols[0],self.symbols[1]) if self.first_long else (self.symbols[1],self.symbols[0])
        self.not_work_pos[s_l] = self.hedge_pos
        self.not_work_pos[s_s] = -self.hedge_pos
    
    def get_need_pos(self,pos_data):
        new_pos_long,new_pos_short = pos_data
        s_l,s_s = (self.symbols[0],self.symbols[1]) if self.first_long else (self.symbols[1],self.symbols[0])
        need_pos = {}
        if self.keep_start_long:
            if new_pos_long == 0:
                self.keep_start_long = False
            else:
                new_pos_long = None
        if self.keep_start_short:
            if new_pos_short == 0:
                self.keep_start_short = False
            else:
                new_pos_short = None
        need_pos[s_l] = new_pos_long
        need_pos[s_s] = new_pos_short
        return need_pos
    
    def get_pos_on_lvl(self,row):
        if self.dh_lvl:
            if row['close'] < self.dh_lvl:
                self.in_work = False if self.keep_hedge else True
                return (self.hedge_pos,-self.hedge_pos)
        if self.uh_lvl:
            if row['close'] > self.uh_lvl:
                self.in_work = False if self.keep_hedge else True
                return (self.hedge_pos,-self.hedge_pos)
        if row['close'] <= self.start_lvl:
            return 1,0
        elif row['close'] >= self.end_lvl:
            return 0,-1
        return None,None
    
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
            pos_data = self.get_pos_on_lvl(row)
            self.need_pos = self.get_need_pos(pos_data)
        else:
            self.need_pos = self.not_work_pos
        return self.need_pos

# TODO start work 30.12.25
class LWS9_(WSBase):
    """ползающая пара"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'start':2500,
            'step_per':0.5, #%
            'first_long':False,
            'amount_touch':2
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.amount_touch = parameters.get('amount_touch',2)
        self.first_long = parameters.get('first_long',False)
        self.start = parameters['start']
        self.step = parameters.get('step_per',1) *  self.start / 100
        self.base_lvl = self.start
        self.get_help_lvls()
        self.change_range = False
        self.default_touch()

    def default_touch(self):
        self.touch_lvls = {
            self.base_lvl : [0,True],
            self.top_lvl : [0,True],
            self.bot_lvl : [0,True],
        }

    def reset_touch(self,name):
        for lvl in self.touch_lvls:
            if lvl != name:
                self.touch_lvls[lvl][1] = True
            else:
                self.touch_lvls[lvl][1] = False
                
    def get_help_lvls(self):
        self.top_lvl = self.base_lvl + self.step
        self.bot_lvl = self.base_lvl - self.step
        self.up_lvl = self.top_lvl + self.step
        self.down_lvl = self.bot_lvl - self.step

    def get_lvls(self,price):
        self.base_lvl = ((price - self.start) // self.step)*self.step + self.start
        self.get_help_lvls()
        self.default_touch()
        # print(self.base_lvl,self.top_lvl,self.bot_lvl,self.up_lvl,self.down_lvl)

    def set_touch(self,name):
        if self.touch_lvls[name][1]:
            self.touch_lvls[name][0] += 1
            self.reset_touch(name)
        # print(self.touch_lvls)
        if self.touch_lvls[name][0] < self.amount_touch:
            return False
        return True

    def get_poss(self,price):
        need_pos = {}
        calc_pos = True
        new_long_pos,new_short_pos = None,None
        s_l,s_s = (self.symbols[0],self.symbols[1]) if self.first_long else (self.symbols[1],self.symbols[0])
        if self.change_range:
            cur_pos_l = abs(self.positions[s_l])
            cur_pos_s = abs(self.positions[s_s])
            if cur_pos_l != cur_pos_s:
                new_long_pos,new_short_pos = 1, -1
                calc_pos = False
            else:
                self.change_range = False
                
        if calc_pos:
            if price > self.up_lvl or price < self.down_lvl:
                new_long_pos,new_short_pos = 1, -1
                self.get_lvls(price)
                self.change_range = True
            elif price > self.top_lvl:
                if self.set_touch(self.top_lvl):
                    new_long_pos = 0
            elif price < self.bot_lvl:
                if self.set_touch(self.bot_lvl):
                    new_short_pos = 0
            elif price > self.base_lvl:
                if self.set_touch(self.base_lvl):
                    new_short_pos = -1
            elif price < self.base_lvl:
                if self.set_touch(self.base_lvl):
                    new_long_pos = 1
        # print(self.base_lvl,price,new_long_pos,new_short_pos,self.positions[s_l],self.positions[s_s])
        need_pos[s_l] = new_long_pos
        need_pos[s_s] = new_short_pos
        return need_pos

    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            self.last_dfs[tf1][s] = df
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        s1 = self.symbols[0]
        row = self.last_dfs[tf1][s1].iloc[-1]
        self.need_pos = self.get_poss(row['close'])
        return self.need_pos