import pandas as pd
import numpy as np


def add_vangerchik(df: pd.DataFrame,period:int,divider:int):
    """
    Добавляет колонки 'max_vg' и 'min_vg' в DataFrame.
    Оптимизированная версия с использованием векторизованных операций.
    
    :param df: DataFrame 
    :return: DataFrame с добавленными колонками 'max_vg', 'min_vg'
    """
    # Верхняя полоса (максимум за последние N периодов)
    df['max_hb'] = df['high'].rolling(window=period).max()
    
    # Нижняя полоса (минимум за последние N периодов)
    df['min_hb'] = df['low'].rolling(window=period).min()
    
    # Средняя линия
    df['avarege'] = (df['max_hb'] + df['min_hb']) / 2
    # Вычисляем разницу между 'max_hb' и 'min_hb'
    diff = df['max_hb'] - df['min_hb']
    step = diff / divider
    # Вычисляем 'max_vg' и 'min_vg' с использованием векторизованных операций
    df['max_vg'] = df['max_hb'] - step
    df['min_vg'] = df['min_hb'] + step
    
    return df