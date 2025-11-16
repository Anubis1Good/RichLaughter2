import pandas as pd
import numpy as np


def add_donchan_channel(df, period=20):
    """
    '''add "max_hb", "min_hb", "average"'''
    
    :param df: DataFrame с колонками 'high', 'low'
    :param period: Период для расчета канала Дончиана (по умолчанию 20)
    :return: DataFrame с добавленными колонками
    """
    # Верхняя полоса (максимум за последние N периодов)
    df['max_hb'] = df['high'].rolling(window=period).max()
    
    # Нижняя полоса (минимум за последние N периодов)
    df['min_hb'] = df['low'].rolling(window=period).min()
    
    # Средняя линия
    df['average'] = (df['max_hb'] + df['min_hb']) / 2
    
    return df

def add_sma(df: pd.DataFrame, period=20, kind='close'):
    """
    Добавляет колонку 'sma' в DataFrame.
    Оптимизированная версия с использованием встроенных функций Pandas.
    
    :param df: DataFrame с колонками 'high', 'low', 'close'
    :param period: Период для SMA (по умолчанию 20)
    :param kind: Тип цены для расчета SMA ('middle', 'high', 'low', 'close')
    :return: DataFrame с добавленной колонкой 'sma'
    """
    
    # Вычисляем SMA с использованием встроенной функции rolling
    df['sma'] = df[kind].rolling(window=period).mean()
    
    return df

def add_bollinger(df: pd.DataFrame, period=20, kind='close', multiplier=2):
    """
    Добавляет колонки 'bbu', 'bbd', 'sma' в DataFrame.
    Оптимизированная версия с использованием встроенных функций Pandas.
    
    :param df: DataFrame с колонками 'high', 'low', 'close'
    :param period: Период для расчета полос Боллинджера (по умолчанию 20)
    :param kind: Тип цены для расчета ('middle', 'high', 'low', 'close')
    :param multiplier: Множитель для стандартного отклонения (по умолчанию 2)
    :return: DataFrame с добавленными колонками 'bbu', 'bbd', 'sma'
    """
    price = df[kind]
    
    # Вычисляем SMA
    df['sma'] = price.rolling(window=period).mean()
    
    # Вычисляем стандартное отклонение
    std_dev = price.rolling(window=period).std()
    
    # Вычисляем верхнюю и нижнюю полосы Боллинджера
    df['bbu'] = df['sma'] + (multiplier * std_dev)
    df['bbd'] = df['sma'] - (multiplier * std_dev)
    
    return df

def add_fractals(df:pd.DataFrame, period=5):
    """
    add 'fractal_up','fractal_down'\n
    Добавляет фракталы Билла Вильямса в DataFrame с данными свечей.
    
    :param df: DataFrame с колонками 'High' и 'Low'
    :param period: Количество свечей для поиска фракталов (по умолчанию 5)
    :return: DataFrame с добавленными колонками 'Fractal_Up' и 'Fractal_Down'
    """
    # Вычисляем смещение для сравнения свечей
    shift = (period - 1) // 2  # Для периода 5 это будет 2
    
    # Фрактал вверх (верхний фрактал)
    fractal_up_condition = True
    for i in range(1, shift + 1):
        fractal_up_condition &= (df['high'] > df['high'].shift(i))
        fractal_up_condition &= (df['high'] > df['high'].shift(-i))
    df['fractal_up'] = fractal_up_condition
    
    # Фрактал вниз (нижний фрактал)
    fractal_down_condition = True
    for i in range(1, shift + 1):
        fractal_down_condition &= (df['low'] < df['low'].shift(i))
        fractal_down_condition &= (df['low'] < df['low'].shift(-i))
    df['fractal_down'] = fractal_down_condition
    
    return df

def add_rsi(df:pd.DataFrame, period=14,kind='close'):
    """
    add 'rsi'\n
    Вычисляет RSI для DataFrame с данными о ценах.
    
    :param data: DataFrame с колонкой 'Close' (цены закрытия)
    :param period: Период RSI (по умолчанию 14)
    :return: DataFrame с добавленной колонкой 'RSI'
    """
    # Вычисляем изменение цены
    delta = df[kind].diff()
    
    # Разделяем на рост и падение
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Вычисляем относительную силу (RS)
    rs = gain / loss
    
    # Вычисляем RSI
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def add_rsi_tw(df:pd.DataFrame, period=14, kind='close'):
    """
    Добавляет колонку 'rsi_tw' в DataFrame с данными о ценах.
    
    :param df: DataFrame с колонкой 'close' (цены закрытия)
    :param period: Период RSI (по умолчанию 14)
    :param kind: Название колонки с ценами (по умолчанию 'close')
    :return: DataFrame с добавленной колонкой 'RSI'
    """
    # Вычисляем изменение цены
    delta = df[kind].diff()
    
    # Разделяем на рост и падение
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Вычисляем экспоненциальное скользящее среднее (EMA) для роста и падения
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Вычисляем относительную силу (RS)
    rs = avg_gain / avg_loss
    
    # Вычисляем RSI
    df['rsi_tw'] = 100 - (100 / (1 + rs))
    
    return df

def add_ema(df:pd.DataFrame, period=20, kind='close'):
    """
    Вычисляет EMA для DataFrame с данными о ценах.
    
    :param df: DataFrame с колонкой цен (по умолчанию 'close')
    :param period: Период EMA (по умолчанию 20)
    :param kind: Название колонки с ценами (по умолчанию 'close')
    :return: DataFrame с добавленной колонкой 'ema'
    """
    alpha = 2 / (period + 1)
    
    # Используем expanding() и apply() для векторизованного расчета EMA
    sma = df[kind].rolling(window=period, min_periods=period).mean()
    ema = df[kind].ewm(alpha=alpha, adjust=False).mean()
    
    # Комбинируем SMA (первые period-1 значений) и EMA (остальные значения)
    df['ema'] = sma.where(sma.notna(), ema)
    
    return df

def add_stochastic(df:pd.DataFrame, k_period=14, d_period=3,kind='close'):
    """add 'lowest_so','highest_so','%k','%d' """
    df['lowest_so'] = df[kind].rolling(window=k_period).min()
    df['highest_so'] = df[kind].rolling(window=k_period).max()
    df['%k'] = 100 * ((df[kind] - df['lowest_so']) / (df['highest_so'] - df['lowest_so']))
    df['%d'] = df['%k'].rolling(window=d_period).mean()
    return df

def add_atr(df:pd.DataFrame, period=5,kind='close'):
    '''"atr"'''
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df[kind].shift(1))
    df['low_close'] = np.abs(df['low'] - df[kind].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df

def add_macd(data:pd.DataFrame, short_window=12, long_window=26, signal_window=9):
    """add 'ema_1','ema_2','macd','signal_line'"""
    data['ema_1'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['ema_2'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['macd'] = data['ema_1'] - data['ema_2']
    data['signal_line'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
    return data

def add_adx(df:pd.DataFrame,adx_period=14):
    """
    'adx'
    Расчет индикатора ADX (Average Directional Index).
    :param df: DataFrame с данными
    :return: DataFrame с добавленным столбцом ADX
    """
    # Расчет True Range (TR)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )

    # Расчет Positive Directional Movement (+DM) и Negative Directional Movement (-DM)
    df['plus_dm'] = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0),
        0
    )
    df['minus_dm'] = np.where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
        np.maximum(df['low'].shift(1) - df['low'], 0),
        0
    )

    # Сглаживание TR, +DM, -DM
    df['tr_smooth'] = df['tr'].rolling(window=adx_period, min_periods=adx_period).sum()
    df['plus_dm_smooth'] = df['plus_dm'].rolling(window=adx_period, min_periods=adx_period).sum()
    df['minus_dm_smooth'] = df['minus_dm'].rolling(window=adx_period, min_periods=adx_period).sum()

    # Расчет +DI и -DI
    df['plus_di'] = (df['plus_dm_smooth'] / df['tr_smooth']) * 100
    df['minus_di'] = (df['minus_dm_smooth'] / df['tr_smooth']) * 100

    # Расчет ADX
    df['dx'] = (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
    df['adx'] = df['dx'].rolling(window=adx_period, min_periods=adx_period).mean()

    return df

def add_kama(df:pd.DataFrame, period=30,fast_ema=2,slow_ema=30):
    """
    Расчет индикатора KAMA (Kaufman Adaptive Moving Average).
    :param df: DataFrame с данными
    :param period: Период для расчета KAMA
    :return: DataFrame с добавленным столбцом KAMA
    """
    change = abs(df['close'] - df['close'].shift(period))
    volatility = df['close'].diff().abs().rolling(window=period).sum()
    efficiency_ratio = change / volatility

    fast_sc = 2 / (fast_ema + 1)
    slow_sc = 2 / (slow_ema + 1)
    smooth_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2

    df[f'kama_{period}'] = 0.0
    for i in range(period, len(df)):
        df.loc[df.index[i], f'kama_{period}'] = (
            df.loc[df.index[i - 1], f'kama_{period}'] +
            smooth_constant[i] * (df.loc[df.index[i], 'close'] - df.loc[df.index[i - 1], f'kama_{period}'])
        )
    return df

def add_chop(df:pd.DataFrame,chop_period=14):
    """
    'chop'
    Расчет индикатора CHOP (Choppiness Index).
    :param df: DataFrame с данными
    :return: DataFrame с добавленным столбцом CHOP
    """
    # Расчет True Range (TR)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )

    # Сумма TR за период
    df['tr_sum'] = df['tr'].rolling(window=chop_period).sum()

    # Максимальная и минимальная цена за период
    df['high_max'] = df['high'].rolling(window=chop_period).max()
    df['low_min'] = df['low'].rolling(window=chop_period).min()

    # Расчет CHOP
    df['chop'] = 100 * np.log10(df['tr_sum'] / (df['high_max'] - df['low_min'])) / np.log10(chop_period)
    return df

def add_cci(df:pd.DataFrame, period=20, kind='close'):
    """
    Добавляет колонку 'cci' в DataFrame с данными о ценах.
    
    :param df: DataFrame с колонкой 'close' (цены закрытия)
    :param period: Период CCI (по умолчанию 20)
    :param kind: Название колонки с ценами (по умолчанию 'close')
    :return: DataFrame с добавленной колонкой 'CCI'
    """
    # Вычисляем типичную цену (Typical Price)
    typical_price = (df['high'] + df['low'] + df[kind]) / 3
    
    # Вычисляем скользящее среднее типичной цены (SMA)
    sma = typical_price.rolling(window=period).mean()
    
    # Вычисляем среднее отклонение (Mean Deviation)
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    
    # Вычисляем CCI
    df['cci'] = (typical_price - sma) / (0.015 * mean_deviation)
    
    return df

def add_williams_r(df:pd.DataFrame, period=14, kind='close'):
    """
    Добавляет колонку 'williams_r' в DataFrame с данными о ценах.
    
    :param df: DataFrame с колонками 'high', 'low' и 'close'
    :param period: Период Williams %R (по умолчанию 14)
    :param kind: Название колонки с ценами закрытия (по умолчанию 'close')
    :return: DataFrame с добавленной колонкой 'williams_r'
    """
    # Вычисляем максимум и минимум за период
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    
    # Вычисляем Williams %R
    df['williams_r'] = -100 * (highest_high - df[kind]) / (highest_high - lowest_low)
    
    return df

def add_mfi(df:pd.DataFrame, period=14):
    """
    Добавляет колонку 'mfi' в DataFrame с данными о ценах и объемах.
    
    :param df: DataFrame с колонками 'high', 'low', 'close' и 'volume'
    :param period: Период MFI (по умолчанию 14)
    :return: DataFrame с добавленной колонкой 'MFI'
    """
    # Вычисляем типичную цену (Typical Price)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Вычисляем денежный поток (Money Flow)
    money_flow = typical_price * df['volume']
    
    # Определяем положительный и отрицательный денежный поток
    positive_flow = (typical_price > typical_price.shift(1)) * money_flow
    negative_flow = (typical_price < typical_price.shift(1)) * money_flow
    
    # Вычисляем сумму положительного и отрицательного денежного потока за период
    positive_flow_sum = positive_flow.rolling(window=period).sum()
    negative_flow_sum = negative_flow.rolling(window=period).sum()
    
    # Вычисляем Money Flow Ratio (MFR)
    money_flow_ratio = positive_flow_sum / negative_flow_sum
    
    # Вычисляем MFI
    df['mfi'] = 100 - (100 / (1 + money_flow_ratio))
    
    return df

def add_awesome_oscillator(df:pd.DataFrame, short_period=5, long_period=34):
    """
    Добавляет колонку 'ao' в DataFrame с данными о ценах.
    
    :param df: DataFrame с колонкой 'close' (цены закрытия)
    :param short_period: Период короткой скользящей средней (по умолчанию 5)
    :param long_period: Период длинной скользящей средней (по умолчанию 34)
    :return: DataFrame с добавленной колонкой 'ao'
    """
    # Вычисляем типичную цену (Typical Price)
    typical_price = (df['high'] + df['low']) / 2
    
    # Вычисляем короткую и длинную скользящие средние (SMA)
    sma_short = typical_price.rolling(window=short_period).mean()
    sma_long = typical_price.rolling(window=long_period).mean()
    
    # Вычисляем Awesome Oscillator (AO)
    df['ao'] = sma_short - sma_long
        # Вычисляем Awesome Oscillator (AO)
    # ao = sma_short - sma_long
    
    # # Делим AO на цену закрытия
    # ao_relative_to_close = ao / df['close']
    
    # df['ao'] = ao_relative_to_close
    
    return df

def add_roc(df:pd.DataFrame, period=12, kind='close'):
    """
    Добавляет колонку 'roc' в DataFrame с данными о ценах.
    
    :param df: DataFrame с колонкой 'close' (цены закрытия)
    :param period: Период ROC (по умолчанию 12)
    :param kind: Название колонки с ценами (по умолчанию 'close')
    :return: DataFrame с добавленной колонкой 'ROC'
    """
    # Вычисляем ROC
    df['roc'] = ((df[kind] - df[kind].shift(period)) / df[kind].shift(period)) * 100
    
    return df

def add_ultimate_oscillator(df:pd.DataFrame, short_period=7, medium_period=14, long_period=28):
    """
    Добавляет колонку 'ultimate_oscillator' в DataFrame с данными о ценах.
    
    :param df: DataFrame с колонками 'high', 'low', 'close'
    :param short_period: Короткий период (по умолчанию 7)
    :param medium_period: Средний период (по умолчанию 14)
    :param long_period: Длинный период (по умолчанию 28)
    :return: DataFrame с добавленной колонкой 'ultimate_oscillator'
    """
    # Вычисляем типичную цену (Typical Price)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Вычисляем денежный поток (Money Flow)
    money_flow = typical_price * df['volume']
    
    # Определяем давление покупок и продаж
    buying_pressure = typical_price - df[['low', 'close']].min(axis=1)
    true_range = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
    
    # Вычисляем средние значения для каждого периода
    avg_buying_pressure_short = buying_pressure.rolling(window=short_period).sum()
    avg_true_range_short = true_range.rolling(window=short_period).sum()
    
    avg_buying_pressure_medium = buying_pressure.rolling(window=medium_period).sum()
    avg_true_range_medium = true_range.rolling(window=medium_period).sum()
    
    avg_buying_pressure_long = buying_pressure.rolling(window=long_period).sum()
    avg_true_range_long = true_range.rolling(window=long_period).sum()
    
    # Вычисляем компоненты осциллятора
    short_component = avg_buying_pressure_short / avg_true_range_short
    medium_component = avg_buying_pressure_medium / avg_true_range_medium
    long_component = avg_buying_pressure_long / avg_true_range_long
    
    # Вычисляем Ultimate Oscillator
    df['ultimate_oscillator'] = (4 * short_component + 2 * medium_component + long_component) / 7 * 100
    
    return df

def add_cmo(df:pd.DataFrame, period=14, kind='close'):
    """
    Добавляет колонку 'cmo' в DataFrame с данными о ценах.
    
    :param df: DataFrame с колонкой 'close' (цены закрытия)
    :param period: Период CMO (по умолчанию 14)
    :param kind: Название колонки с ценами (по умолчанию 'close')
    :return: DataFrame с добавленной колонкой 'CMO'
    """
    # Вычисляем изменение цены
    delta = df[kind].diff()
    
    # Разделяем на рост и падение
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Вычисляем сумму роста и падения за период
    sum_gain = gain.rolling(window=period).sum()
    sum_loss = loss.rolling(window=period).sum()
    
    # Вычисляем CMO
    df['cmo'] = ((sum_gain - sum_loss) / (sum_gain + sum_loss)) * 100
    
    return df


def add_keltner_channel(df:pd.DataFrame, period=20, multiplier=2):
    """
    Добавляет колонки 'keltner_upper', 'keltner_middle', 'keltner_lower' в DataFrame.
    
    :param df: DataFrame с колонками 'high', 'low', 'close'
    :param period: Период для EMA и ATR (по умолчанию 20)
    :param multiplier: Множитель для ATR (по умолчанию 2)
    :return: DataFrame с добавленными колонками
    """
    # Вычисляем EMA (центральная линия)
    df['keltner_middle'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Вычисляем ATR (средний истинный диапазон)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    # Вычисляем верхнюю и нижнюю полосы
    df['keltner_upper'] = df['keltner_middle'] + (multiplier * atr)
    df['keltner_lower'] = df['keltner_middle'] - (multiplier * atr)
    
    return df