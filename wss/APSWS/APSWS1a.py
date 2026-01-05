import pandas as pd
from wss.WSBase import WSBase
from indicators.arbitration_ind import get_percent_diff_window

class APSWS1_DYNAMO(WSBase):
    """сводный арбитраж"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'first_long': True,
            'funding': False,
            'hour_fund':18,
            'minute_fund':24
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.first_long = parameters.get('first_long',False)
        self.funding = parameters.get('funding',False)
        self.hour_fund = parameters.get('hour_fund',18)
        self.minute_fund = parameters.get('minute_fund',20)
    
    def get_need_pos(self,row):
        s_l,s_s = (self.symbols[0],self.symbols[1]) if self.first_long else (self.symbols[1],self.symbols[0])
        need_pos = {}
        if row['weekday'] < 5 and self.funding:
            if row['hour'] == self.hour_fund and row['minute'] > self.minute_fund:
                s_l,s_s = s_s,s_l
        need_pos[s_l] = 1
        need_pos[s_s] = -1
        return need_pos
    
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            df['ms'] = pd.to_datetime(df['ms'])
            df['hour'] = df['ms'].dt.hour  
            df['minute'] = df['ms'].dt.minute
            df['weekday'] = df['ms'].dt.weekday
            self.last_dfs[tf1][s] = df
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        s1 = self.symbols[0]
        row = self.last_dfs[tf1][s1].iloc[-1]
        self.need_pos = self.get_need_pos(row)

        return self.need_pos
    
# 30.12.2025 
class APSWS2_SPARTACUS(WSBase):
    """статистический арбитраж"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'min_desc':0.2,
            'buff_0':0.05,
            'window':145,
            'smooth':None,
            'kind':'close',
            'reverse_pos':False,
            'funding': 0, # 0 | 1 | -1 | 'close_all'
            'hour_fund':18,
            'minute_fund':24,

        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.funding = parameters.get('funding',False)
        self.hour_fund = parameters.get('hour_fund',18)
        self.minute_fund = parameters.get('minute_fund',20)
        self.min_desc = parameters.get('min_desc',0.5)
        self.window = parameters.get('window',100)
        self.smooth = parameters.get('smooth',None)
        self.kind = parameters.get('kind','close')
        self.buff_0 = parameters.get('buff_0',0.05)
        self.reverse_pos = parameters.get('reverse_pos',False)
        self.enter_desc = 0

    def get_need_pos(self,row):
        s_1,s_2 = self.symbols[0],self.symbols[1]
        need_pos = {}
        pos_s1,pos_s2 = None,None
        if pd.notna(row['desc']):
            if row['desc'] > self.min_desc:
                pos_s1,pos_s2 = -1,1
                self.enter_desc = 1
            elif row['desc'] < -self.min_desc:
                pos_s1,pos_s2 = 1,-1
                self.enter_desc = -1
            elif self.reverse_pos:
                pos_s1,pos_s2 = None,None
            elif self.enter_desc > 0:
                if row['desc'] < self.buff_0:
                    pos_s1,pos_s2 = 0,0
                    self.enter_desc = 0
            elif self.enter_desc < 0:
                if row['desc'] > -self.buff_0:
                    pos_s1,pos_s2 = 0,0
                    self.enter_desc = 0
        if row['weekday'] < 5 and self.funding:
            if row['hour'] == self.hour_fund and row['minute'] > self.minute_fund:
                if self.funding == 'close_all':
                    pos_s1,pos_s2 = 0, 0
                else:
                    pos_s1,pos_s2 = (-1, 1) if self.funding > 0 else (1,-1)
        if abs(self.positions[s_1]) != abs(self.positions[s_2]):
            if pos_s1 is None:
                if abs(self.positions[s_1]) > abs(self.positions[s_2]):
                    sing_s1 = 1 if self.positions[s_1] > 0 else -1
                    pos_s1,pos_s2 = sing_s1, -sing_s1
                else:
                    sing_s2 = 1 if self.positions[s_2] > 0 else -1
                    pos_s1,pos_s2 = sing_s2, -sing_s2
        need_pos[s_1] = pos_s1
        need_pos[s_2] = pos_s2
        # print(pos_s1,pos_s2,row['desc'])
        return need_pos
    
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            df['ms'] = pd.to_datetime(df['ms'])
            df['hour'] = df['ms'].dt.hour  
            df['minute'] = df['ms'].dt.minute
            df['weekday'] = df['ms'].dt.weekday
            self.last_dfs[tf1][s] = df
        self.last_dfs[tf1][self.symbols[0]]['desc'] = get_percent_diff_window( self.last_dfs[tf1][self.symbols[0]],self.last_dfs[tf1][self.symbols[1]],self.window,self.smooth,self.kind)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        s1 = self.symbols[0]
        row = self.last_dfs[tf1][s1].iloc[-1]
        # print(row)
        self.need_pos = self.get_need_pos(row)
        return self.need_pos

# 03.01.2026 
class APSWS3_PROMETHEUS(WSBase):
    """сеточный статистический арбитраж"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'desc_lvls':(0.05,0.1,0.2),
            'buff_0':0.01,
            'window':145,
            'smooth':None,
            'kind':'close',
            'reverse_pos':False,
            'funding': 0, # 0 | 1 | -1 | 'close_all'
            'hour_fund':18,
            'minute_fund':24,
            'keep_pos':False

        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.funding = parameters.get('funding',False)
        self.hour_fund = parameters.get('hour_fund',18)
        self.minute_fund = parameters.get('minute_fund',20)
        self.desc_lvls = parameters.get('desc_lvls',(0.1,0.2))
        self.window = parameters.get('window',100)
        self.smooth = parameters.get('smooth',None)
        self.kind = parameters.get('kind','close')
        self.buff_0 = parameters.get('buff_0',0.05)
        self.reverse_pos = parameters.get('reverse_pos',False)
        self.keep_pos = parameters.get('keep_pos',False)
        self.enter_desc = 0

    def get_need_pos(self,row):
        s_1,s_2 = self.symbols[0],self.symbols[1]
        need_pos = {}
        pos_s1,pos_s2 = None,None
        max_pos = 1
        if row['desc'] > 0:
            enter_desc = 1
        else:
            enter_desc = -1
        for lvl in self.desc_lvls:
            if row['desc'] > lvl or row['desc'] < -lvl:
                max_pos += 1
        if max_pos != 1:
            new_pos = max_pos -1 
            if enter_desc == 1:
                if self.positions[s_1] <= -max_pos:
                    if self.keep_pos:
                        pos_s1,pos_s2 = None,None
                    elif self.reverse_pos:
                        if self.positions[s_1] < 0:
                            pos_s1,pos_s2 = None,None
                        else:
                            pos_s1,pos_s2 = -new_pos,new_pos
                    else:
                        pos_s1,pos_s2 = -max_pos,max_pos
                else:
                    pos_s1,pos_s2 = -new_pos,new_pos
            else:
                if self.positions[s_1] >= max_pos:
                    if self.keep_pos:
                        pos_s1,pos_s2 = None,None
                    elif self.reverse_pos:
                        if self.positions[s_1] > 0:
                            pos_s1,pos_s2 = None,None
                        else:
                            pos_s1,pos_s2 = new_pos,-new_pos
                    else:
                        pos_s1,pos_s2 = max_pos,-max_pos
                else:
                    pos_s1,pos_s2 = new_pos,-new_pos
            self.enter_desc = enter_desc
        if not self.reverse_pos:
            if self.enter_desc > 0:
                if row['desc'] < self.buff_0:
                    pos_s1,pos_s2 = 0,0
                    self.enter_desc = 0
            elif self.enter_desc < 0:
                if row['desc'] > -self.buff_0:
                    pos_s1,pos_s2 = 0,0
                    self.enter_desc = 0
        if row['weekday'] < 5 and self.funding:
            if row['hour'] == self.hour_fund and row['minute'] > self.minute_fund:
                if self.funding == 'close_all':
                    pos_s1,pos_s2 = 0, 0
                else:
                    pos_s1,pos_s2 = (-1, 1) if self.funding > 0 else (1,-1)
        if abs(self.positions[s_1]) != abs(self.positions[s_2]):
            if pos_s1 is None:
                if abs(self.positions[s_1]) > abs(self.positions[s_2]):
                    sing_s1 = 1 if self.positions[s_1] > 0 else -1
                    pos_s1,pos_s2 = sing_s1, -sing_s1
                else:
                    sing_s2 = 1 if self.positions[s_2] > 0 else -1
                    pos_s1,pos_s2 = sing_s2, -sing_s2
        need_pos[s_1] = pos_s1
        need_pos[s_2] = pos_s2
        # print(pos_s1,pos_s2,row['desc'])
        return need_pos
    
    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            df['ms'] = pd.to_datetime(df['ms'])
            df['hour'] = df['ms'].dt.hour  
            df['minute'] = df['ms'].dt.minute
            df['weekday'] = df['ms'].dt.weekday
            self.last_dfs[tf1][s] = df
        self.last_dfs[tf1][self.symbols[0]]['desc'] = get_percent_diff_window( self.last_dfs[tf1][self.symbols[0]],self.last_dfs[tf1][self.symbols[1]],self.window,self.smooth,self.kind)
        return self.last_dfs
    
    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        s1 = self.symbols[0]
        row = self.last_dfs[tf1][s1].iloc[-1]
        self.need_pos = self.get_need_pos(row)

        return self.need_pos