import pandas as pd
from wss.WSBase import WSBase

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