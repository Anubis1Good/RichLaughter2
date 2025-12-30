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
            'minute_fund':20
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
class APSWS2_(WSBase):
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
            'minute_fund':20,

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