import pandas as pd
import numpy as np

def add_volume_profile(df:pd.DataFrame, period=14):
    """
    Добавляет Volume Profile в DataFrame.
    
    :param df: DataFrame с колонками 'high', 'low', 'close', 'volume'
    :param period: Период для расчета Volume Profile (по умолчанию 14)
    :return: DataFrame с добавленными колонками 'poc', 'value_area_high', 'value_area_low'
    """
    # Создаем пустые колонки для результатов
    df['poc'] = np.nan
    df['value_area_high'] = np.nan
    df['value_area_low'] = np.nan
    
    for i in range(period, len(df)):
        # Выбираем данные за последние `period` дней
        window = df.iloc[i-period:i]
        
        # Создаем гистограмму объема по ценам
        price_bins = np.linspace(window['low'].min(), window['high'].max(), num=100)
        volume_profile = np.zeros_like(price_bins)
        
        for j in range(len(window)):
            low = window.iloc[j]['low']
            high = window.iloc[j]['high']
            close = window.iloc[j]['close']
            volume = window.iloc[j]['volume']
            
            # Распределяем объем по ценам
            mask = (price_bins >= low) & (price_bins <= high)
            volume_profile[mask] += volume
        
        # Находим POC (цена с максимальным объемом)
        poc_index = np.argmax(volume_profile)
        poc = price_bins[poc_index]
        
        # Находим Value Area (70% объема)
        total_volume = np.sum(volume_profile)
        sorted_volume_indices = np.argsort(volume_profile)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_volume_indices:
            cumulative_volume += volume_profile[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= 0.7 * total_volume:
                break
        
        value_area_prices = price_bins[value_area_indices]
        value_area_high = np.max(value_area_prices)
        value_area_low = np.min(value_area_prices)
        
        # Записываем результаты
        df.at[df.index[i], 'poc'] = poc
        df.at[df.index[i], 'value_area_high'] = value_area_high
        df.at[df.index[i], 'value_area_low'] = value_area_low
    
    return df