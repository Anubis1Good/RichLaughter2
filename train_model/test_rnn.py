import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Загрузка данных
df = pd.read_parquet('data_for_tests\data_from_moex5\_5IMOEXF_1_1763893692.parquet')

# Выбираем признаки для обучения
features = ['open', 'high', 'low', 'vol_coin', 'volume', 'direction', 
           'middle']
target = 'close'