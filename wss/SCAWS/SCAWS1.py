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