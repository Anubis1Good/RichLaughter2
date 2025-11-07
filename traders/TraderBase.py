import pandas as pd
class TraderBase:
    def __init__(self):
        pass

    def start_info(self):
        print('ApiBase')

    def _get_balance(self):
        pass
    
    def _check_risk(self):
        pass

    def _check_time(self):
        pass

    def _check_position(self):
        pass

    def _check_req(self):
        pass

    def _send_open(self,direction,quantity):
        pass

    def _send_close(self,direction,quantity):
        pass

    def _reverse_pos(self,direction):
        pass
    
    def _close_all_pos(self):
        pass

    def _reset_req(self):
        pass

    def _get_df(self) -> pd.DataFrame:
        df = pd.DataFrame()
        return df
    
    def _debug_log(self):
        pass
    
    def _work_ws(self,new_pos,pos):
        pass

    def run(self):
        pass