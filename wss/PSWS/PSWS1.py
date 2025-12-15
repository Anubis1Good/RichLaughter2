import pandas as pd
from wss.WSBase import WSBase
from indicators.classic_ind import add_fractals

# TODO
class PSWS1_(WSBase):
    """Сетка по фракталам"""
    def __init__(self, symbols, timeframes, positions, middle_price, parameters):
        """
        parameters = {
            'period_fractals':10,
        }
        """
        super().__init__(symbols, timeframes, positions, middle_price, parameters)
        self.period_fractals = parameters['period_fractals']
    
    def preprocessing(self, dfs, poss):
        self.last_dfs = {}
        self.update_poss_mps(poss)
        for t in dfs:
            self.last_dfs[t] = {}
            for s in dfs[t]:
                df:pd.DataFrame = dfs[t][s]
                df = add_fractals(df,self.period_fractals)
                ...
                self.last_dfs[t][s] = df
        return self.last_dfs
    
    # def __call__(self, *args, **kwds):
    #     tf1 = self.timeframes[0]
    #     for s in self.last_dfs[tf1]:
    #         row = self.last_dfs[tf1][s].iloc[-1]   
    #         self.grid_func(row,s)
    #     return self.need_pos