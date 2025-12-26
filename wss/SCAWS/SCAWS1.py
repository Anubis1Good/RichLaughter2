import pandas as pd
from wss.WSBase import WSBase
from indicators.classic_ind import add_bollinger,add_rsi

class SCAWS1_mini(WSBase):
    """стратегия типа STA_mini для RL2"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'period_bb':50,
            'period_rsi':14,
            'long_dir':True,
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)

    # def preprocessing(self, df):
    #     df = add_bollinger(df,self.period)
    #     # df = add_big_volume(df,self.period,3)
    #     # df = add_over_bb(df)
    #     df = add_rsi(df,self.period)
    #     df['sma_delta'] = df['sma'].pct_change()
    #     df['dynamic_sma'] = df['sma_delta'].rolling(self.period).mean()
    #     return df
    # def __call__(self, row, *args, **kwds):
    #     if self.go_long and row['dynamic_sma'] < -0.00001:
    #         return 'close_long_pw'
    #     if not self.go_long and row['dynamic_sma'] > 0.00001:
    #         return 'close_short_pw'
    #     if row['high'] > row['bbu']:
    #         if row['is_big'] or row['over_bbu'] or row['rsi'] > 85:
    #             return 'close_long_pw'
    #     if row['low'] < row['bbd']:
    #         if row['is_big'] or row['over_bbd'] or row['rsi'] < 15:
    #             return 'close_short_pw'
    #     if row['low'] < row['sma'] and self.go_long and row['dynamic_sma'] > 0:
    #         return 'long_pw'
    #     if row['high'] > row['sma']and not self.go_long and row['dynamic_sma'] < 0:
    #         return 'short_pw'

class SCAWS2_SHARKNADO(WSBase):
    """стратегия типа STA_mini для RL2"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'period_sma':50,
            'grid_dir':0,
            'amount_lvl': 3,
            'percent_step': 0.1 #%
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.period_sma = parameters['period_sma']
        self.amount_lvl = parameters['amount_lvl']
        self.percent_step = parameters['percent_step'] * 0.01
        self.long_pos = 1 if parameters['grid_dir'] > -1 else 0
        self.short_pos = -1 if parameters['grid_dir'] < 1 else 0

    def preprocessing(self, dfs, poss):
        self.update_poss_mps(poss)
        tf1 = self.timeframes[0]
        self.last_dfs = {tf1:{}}
        
        for s in dfs[tf1]:
            df = dfs[tf1][s].copy()
            df['sma'] = df['close'].rolling(self.period_sma).mean()
            for i in range(1,self.amount_lvl):
                df['top_'+str(i)] = df['sma'] + df['close'] * i * self.percent_step 
                df['bot_'+str(i)] = df['sma'] - df['close'] * i * self.percent_step
            self.last_dfs[tf1][s] = df
        return self.last_dfs

    def __call__(self, *args, **kwds):
        tf1 = self.timeframes[0]
        for s in self.last_dfs[tf1]:
            row = self.last_dfs[tf1][s].iloc[-1]
            new_pos = 0
            cur_dir = 0
            for col in self.last_dfs[tf1][s].columns.to_list():
                if 'top_' in col:
                    if row['close'] > row[col]:
                        new_pos += self.long_pos
                        cur_dir = -1
                if 'bot_' in col:
                    if row['close'] < row[col]:
                        new_pos += self.short_pos
                        cur_dir = 1
            if new_pos < 0:
                if self.positions[s] == new_pos - 1:
                    new_pos = None
            elif new_pos > 0:
                if self.positions[s] == new_pos + 1:
                    new_pos = None
            elif cur_dir == 0:
                new_pos = None
            self.need_pos[s] = new_pos

        return self.need_pos