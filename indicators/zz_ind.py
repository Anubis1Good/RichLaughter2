import pandas as pd
import numpy as np

def add_dzz_peaks(df: pd.DataFrame, source='high_low', n_std=1.5, method='std', period=20,drop_last=True):
    """
    add 'zigzag','zigzag_peaks'
    ZigZag с динамическим reversal на основе волатильности
    
    Параметры:
    df - DataFrame с колонками: high, low, close
    source - 'high_low' (по экстремумам) или 'close' (по ценам закрытия)
    n_std - множитель для std или среднего (1.5 по умолчанию)
    method - 'std' (стандартное отклонение) или 'mean' (средний диапазон)
    period - период для расчета волатильности
    
    Возвращает:
    df с колонками:
        zigzag - линейно интерполированные значения зигзага
        zigzag_peaks - точки перелома (пики/впадины)
        zigzag_direction - направление (1=up, -1=down)
        reversal_threshold - порог разворота
    """
    df = df.copy()
    
    # Проверка на достаточное количество данных
    if len(df) < period:
        raise ValueError(f"Недостаточно данных. Требуется минимум {period} баров")
    
    # Выбор источника данных
    if source == 'high_low':
        prices = df[['high', 'low']].values
    elif source == 'close':
        prices = df[['close', 'close']].values
    else:
        raise ValueError("source должен быть 'high_low' или 'close'")
    
    highs = prices[:, 0]
    lows = prices[:, 1]
    size = len(df)
    
    # Расчет динамического порога разворота
    if method == 'std':
        rolling_std = df['close'].rolling(period).std().bfill()
        reversal_values = rolling_std * n_std
    elif method == 'mean':
        ranges = df['high'] - df['low']
        reversal_values = ranges.rolling(period).mean().bfill() * n_std
    else:
        raise ValueError("method должен быть 'std' или 'mean'")
    
    # Инициализация массивов
    zz = np.full(size, np.nan)  # Точки разворота
    direction = np.zeros(size, dtype=np.int8)  # 1=up, -1=down
    
    # Начальные условия
    first_valid = max(1, period-1)  # Первый валидный индекс после заполнения rolling
    direction[:first_valid] = 1
    last_pivot = highs[first_valid]
    last_pivot_idx = first_valid
    zz[first_valid] = last_pivot  # Первая точка
    
    for i in range(first_valid+1, size):
        high = highs[i]
        low = lows[i]
        reversal = reversal_values.iloc[i]
        prev_dir = direction[i-1]
        
        if prev_dir == 1:  # Предыдущее направление - вверх
            # Сначала проверяем обновление максимума
            if high > last_pivot:
                # Удаляем старый максимум
                zz[last_pivot_idx] = np.nan
                last_pivot = high
                last_pivot_idx = i
                zz[i] = last_pivot
                direction[i] = 1  # Подтверждаем текущее направление
            # Затем проверяем разворот (только если не обновили максимум)
            elif low <= last_pivot - reversal:
                direction[i] = -1
                last_pivot = low
                last_pivot_idx = i
                zz[i] = last_pivot
            else:
                direction[i] = 1
                
        else:  # Предыдущее направление - вниз
            # Сначала проверяем обновление минимума
            if low < last_pivot:
                # Удаляем старый минимум
                zz[last_pivot_idx] = np.nan
                last_pivot = low
                last_pivot_idx = i
                zz[i] = last_pivot
                direction[i] = -1  # Подтверждаем текущее направление
            # Затем проверяем разворот (только если не обновили минимум)
            elif high >= last_pivot + reversal:
                direction[i] = 1
                last_pivot = high
                last_pivot_idx = i
                zz[i] = last_pivot
            else:
                direction[i] = -1
    
    # Сохраняем точки перелома до интерполяции
    if drop_last:
        zz[-1] = np.nan
    df['zigzag_peaks'] = zz.copy()
    
    # Линейная интерполяция между точками для непрерывного зигзага
    zz_final = np.full(size, np.nan)
    start_idx = None
    start_val = np.nan
    
    for i in range(size):
        if not np.isnan(zz[i]):
            if start_idx is not None:
                zz_final[start_idx:i+1] = np.linspace(start_val, zz[i], i - start_idx + 1)
            start_idx = i
            start_val = zz[i]
    
    df['zigzag'] = zz_final
    df['zigzag_direction'] = direction
    df['reversal_threshold'] = reversal_values
    
    return df

def add_pzz_peaks(df: pd.DataFrame, source='high_low', percent_threshold=0.1, drop_last=True):
    """
    add 'zigzag','zigzag_peaks'
    ZigZag с динамическим reversal на основе процентного отклонения
    
    Параметры:
    df - DataFrame с колонками: high, low, close
    source - 'high_low' (по экстремумам) или 'close' (по ценам закрытия)
    percent_threshold - процент отклонения для разворота (0.1 = 0.1%)
    drop_last - исключать последний бар (еще не сформировавшийся)
    
    Возвращает:
    df с колонками:
        zigzag - линейно интерполированные значения зигзага
        zigzag_peaks - точки перелома (пики/впадины)
        zigzag_direction - направление (1=up, -1=down)
        reversal_threshold - порог разворота (в абсолютных значениях)
    """
    df = df.copy()
    
    # Выбор источника данных
    if source == 'high_low':
        prices = df[['high', 'low']].values
    elif source == 'close':
        prices = df[['close', 'close']].values
    else:
        raise ValueError("source должен быть 'high_low' или 'close'")
    
    highs = prices[:, 0]
    lows = prices[:, 1]
    size = len(df)
    
    # Инициализация массивов
    zz = np.full(size, np.nan)  # Точки разворота
    direction = np.zeros(size, dtype=np.int8)  # 1=up, -1=down
    reversal_values = np.full(size, np.nan)  # Пороги разворота
    
    # Начальные условия
    direction[0] = 1  # Начинаем с восходящего тренда
    last_pivot = highs[0]
    last_pivot_idx = 0
    zz[0] = last_pivot  # Первая точка
    
    for i in range(1, size):
        high = highs[i]
        low = lows[i]
        reversal = last_pivot * (percent_threshold / 100)  # Вычисляем процентный порог
        reversal_values[i] = reversal  # Сохраняем порог
        prev_dir = direction[i-1]
        
        if prev_dir == 1:  # Предыдущее направление - вверх
            # Сначала проверяем обновление максимума
            if high > last_pivot:
                # Удаляем старый максимум
                zz[last_pivot_idx] = np.nan
                last_pivot = high
                last_pivot_idx = i
                zz[i] = last_pivot
                direction[i] = 1  # Подтверждаем текущее направление
            # Затем проверяем разворот (только если не обновили максимум)
            elif low <= last_pivot - reversal:
                direction[i] = -1
                last_pivot = low
                last_pivot_idx = i
                zz[i] = last_pivot
            else:
                direction[i] = 1
                
        else:  # Предыдущее направление - вниз
            # Сначала проверяем обновление минимума
            if low < last_pivot:
                # Удаляем старый минимум
                zz[last_pivot_idx] = np.nan
                last_pivot = low
                last_pivot_idx = i
                zz[i] = last_pivot
                direction[i] = -1  # Подтверждаем текущее направление
            # Затем проверяем разворот (только если не обновили минимум)
            elif high >= last_pivot + reversal:
                direction[i] = 1
                last_pivot = high
                last_pivot_idx = i
                zz[i] = last_pivot
            else:
                direction[i] = -1
    
    # Сохраняем точки перелома до интерполяции
    if drop_last:
        zz[-1] = np.nan
    df['zigzag_peaks'] = zz.copy()
    
    # Линейная интерполяция между точками для непрерывного зигзага
    zz_final = np.full(size, np.nan)
    start_idx = None
    start_val = np.nan
    
    for i in range(size):
        if not np.isnan(zz[i]):
            if start_idx is not None:
                zz_final[start_idx:i+1] = np.linspace(start_val, zz[i], i - start_idx + 1)
            start_idx = i
            start_val = zz[i]
    
    df['zigzag'] = zz_final
    df['zigzag_direction'] = direction
    df['reversal_threshold'] = reversal_values
    
    return df

def add_dzz_level_channel(df:pd.DataFrame):
    """add 'upper_channel','lower_channel'"""
    points = df[~pd.isna(df['zigzag_peaks'])].iloc[:-1]
    df['upper_channel'] = points[points['zigzag_direction'] == 1]['zigzag_peaks']
    df['lower_channel'] = points[points['zigzag_direction'] == -1]['zigzag_peaks']

    df['upper_channel'] = df['upper_channel'].ffill()
    df['lower_channel'] = df['lower_channel'].ffill()
    return df