from loaders.api_moex_loader.api_moex_loader import ApiMoexLoader
from datetime import date,timedelta

today = date.today()
start_date = str(today - timedelta(days=30))
# # start_date = '2025-02-01'

tickers = ('IMOEXF','MMZ5','RMZ5','SRZ5','GAZPF','SBERF','CRZ5','CNYRUBF','SVZ5','GZZ5',)
tickers = ('GLDRUBF','GLH6')

for ticker in tickers:
    loader = ApiMoexLoader(ticker,'RFUD','forts','futures')
    loader.save_df(start_date,timeframe=1,sformat='parquet')

# loader = ApiMoexLoader('IMOEXF','RFUD','forts','futures')
# loader.save_df(start_date,timeframe=1,sformat='parquet')

# loader = ApiMoexLoader('MMZ5','RFUD','forts','futures')
# loader.save_df(start_date,timeframe=1,sformat='parquet')

# loader = ApiMoexLoader('CNYRUBF','RFUD','forts','futures')
# loader.save_df(start_date,timeframe=1,sformat='parquet')

# loader = ApiMoexLoader('CRZ5','RFUD','forts','futures')
# loader.save_df(start_date,timeframe=1,sformat='parquet')
