import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from time import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from traders.TraderBase import TraderBase
from wss.WSBase import WSBase
from utils.processing.add_vtb_fee_fut import get_func_vtb_fee
from utils.df_utils.convert_timeframe import convert_timeframe

def duration_time(func):
    def wrapper(self, *args, **kwargs):
        start = time()
        print('start', func.__name__)
        result = func(self, *args, **kwargs)
        print('Time:', time() - start)
        return result
    return wrapper

class TestTrader(TraderBase):
    def __init__(self, 
                symbols:List[str],
                timeframes:List[str|int],
                quantity:List[int],
                ws:WSBase = (WSBase,dict()),
                sufix_debug='Test1', 
                need_debug=False, 
                charts:Dict[str,str]=dict(), 
                fee=0.0002,
                close_on_time=False,
                close_map=((23,30),(23,30),(23,30),(23,30),(23,30),(17,50),(17,50),)):
        """
        charts = {
        'tf1': {
            's1':'./df1.csv',
            's2':'./df2.csv'
            }
        }
        Важно!!! Таймфреймы передаются от меньшего к большему!
        (1,5,60) ✔
        (60,5,1) ❌
        """
        super().__init__(symbols, timeframes, quantity, ws, sufix_debug, need_debug)
        self.fee = fee # не в процентах, а в долях от 1
        self.init_charts = charts
        self.charts = {tf: {} for tf in timeframes}
        self.read_dfs()
        self.get_total_time_range()
        self.reload_data()
        self.fee_one_p = fee * 100 #в процентах
        self.close_on_time = close_on_time
        self.close_map = close_map
        self.vtb_fee_funcs = {s: get_func_vtb_fee(s) for s in symbols}

    def reload_data(self):
        self.trade_data = {
            symbol : {
                'total':0,
                'count':0, #количество разворотов
                'amount':0, #размер сделок
                'fees': 0, #комиссия в ???
                'total_wfees_per':0, #прибыль в процентах с учетом комиссии
                'equity':[0], #динамика дохода
                'equity_fee':[0], #динамика дохода с комиссией
                # 'equity_vtb':[0], #динамика дохода с комиссией vtb
                'step_eq_fee':[0], #equity каждый шаг
                'step_eq_vtb':[0], #equity каждый шаг
                'pos':0, #текущая позиция
                'mp':0, #текущая цена
                'o_longs':[], #входы в лонг
                'o_shorts':[], #входы в шорт
                'c_longs':[], #закрытие лонгов
                'c_shorts':[], #закрытие шортов
            } for symbol in self.symbols
        }
        self.open_fee = {symbol:0 for symbol in self.symbols}

    def read_dfs(self):
        for t in self.init_charts:
            for s in self.init_charts[t]:
                path_df = self.init_charts[t][s]
                if path_df.endswith('.parquet'):
                    df = pd.read_parquet(path_df)
                else:
                    df = pd.read_csv(path_df)
                self.charts[t][s] = df
    
    def get_total_time_range(self):
        tf1 = self.timeframes[0]
        start_time = None
        end_time = None
        
        # Находим самое позднее время начала и самое раннее время окончания
        for s in self.charts[tf1]:
            df = self.charts[tf1][s]
            
            # Преобразуем столбец ms в datetime, если еще не сделано
            if not pd.api.types.is_datetime64_any_dtype(df['ms']):
                df['ms'] = pd.to_datetime(df['ms'], format='%Y-%m-%d %H:%M:%S')
            
            # Находим минимальное и максимальное время для текущего символа
            current_start = df['ms'].min()
            current_end = df['ms'].max()
            
            # Обновляем общее время начала (берем самое позднее)
            if start_time is None or current_start > start_time:
                start_time = current_start
            
            # Обновляем общее время окончания (берем самое раннее)
            if end_time is None or current_end < end_time:
                end_time = current_end
        self.start_time = start_time
        self.end_time = end_time
        # Теперь нужно проверить все таймфреймы, а не только первый
        if len(self.timeframes) > 1:
            for tf in self.timeframes[1:]:  # начинаем со второго таймфрейма
                for s in self.charts[tf]:
                    df = self.charts[tf][s]
                    
                    if not pd.api.types.is_datetime64_any_dtype(df['ms']):
                        df['ms'] = pd.to_datetime(df['ms'], format='%Y-%m-%d %H:%M:%S')
                    
                    current_start = df['ms'].min()
                    current_end = df['ms'].max()
                    
                    # Обновляем общее время начала
                    if current_start > start_time:
                        start_time = current_start
                    
                    # Обновляем общее время окончания
                    if current_end < end_time:
                        end_time = current_end
        self.all_start_time = start_time
        self.all_end_time = end_time
    
    def get_deep_copy_last_dfs(self):
        # Глубокое копирование всей структуры
        dfs = {}
        for tf in self.ws.last_dfs:
            dfs[tf] = {}
            for s in self.ws.last_dfs[tf]:
                dfs[tf][s] = self.ws.last_dfs[tf][s].copy(deep=True)
        return dfs

    # ==========================================================================

    def work_need_pos(self, need_pos, last_prices, last_xs):
        for s in self.symbols:
            if need_pos[s] is not None: #новая позиция не None
                if need_pos[s] != self.trade_data[s]['pos']: #новая позиция не равна старой
                    self._process_position_change(s, need_pos[s], last_prices[s], last_xs[s])
                    self.trade_data[s]['step_eq_vtb'].append(self.vtb_fee_funcs[s](self.trade_data[s]['equity'][-1],self.trade_data[s]['amount']))
            else:
                self.trade_data[s]['step_eq_vtb'].append(self.trade_data[s]['step_eq_vtb'][-1])
            self.trade_data[s]['step_eq_fee'].append(self.trade_data[s]['equity'][-1])
            self.trade_data[s]['total'] += self.trade_data[s]['equity'][-1]

    def _process_position_change(self, symbol, new_pos, new_price, last_x):
        """Основной метод обработки изменения позиции"""
        old_pos = self.trade_data[symbol]['pos']
        delta_pos = new_pos - old_pos #the delta should be
        if delta_pos > 0:
            self._handle_positive_delta(symbol, new_pos, old_pos, new_price, last_x,delta_pos)
        else:
            self._handle_negative_delta(symbol, new_pos, old_pos, new_price, last_x,delta_pos)
        # vvv ????
        self.trade_data[symbol]['pos'] = new_pos

    def _handle_positive_delta(self, symbol, new_pos, old_pos, new_price, last_x,delta_pos):
        """Обработка увеличения позиции (delta_pos > 0)"""
        if old_pos >= 0:  # add_long или open_long
            self._handle_long_operations(symbol, new_pos, old_pos, new_price, last_x,delta_pos)
        else:  # close_short (pos < 0)
            self._handle_short_closing(symbol, new_pos, old_pos, new_price, last_x,delta_pos)

    def _handle_negative_delta(self, symbol, new_pos, old_pos, new_price, last_x,delta_pos):
        """Обработка уменьшения позиции (delta_pos < 0)"""
        if old_pos <= 0:  # add_short или open_short
            self._handle_short_operations(symbol, new_pos, old_pos, new_price, last_x,delta_pos)
        else:  # close_long (pos > 0)
            self._handle_long_closing(symbol, new_pos, old_pos, new_price, last_x,delta_pos)

    def _handle_long_operations(self, symbol, new_pos, old_pos, new_price, last_x,delta_pos):
        """Обработка операций с лонгами (открытие/добавление)"""
        feei = self.fee * new_price #fee by one
        cur_feei = feei * delta_pos #full fee
        if old_pos == 0:  # open_long
            self.trade_data[symbol]['mp'] = new_price
            self.open_fee[symbol] = cur_feei
        else:  # add_long
            old_price = self.trade_data[symbol]['mp']
            self.trade_data[symbol]['mp'] = (old_price * old_pos + new_price * (new_pos - old_pos)) / new_pos
            self.open_fee[symbol] += cur_feei
        
        self._update_fee_metrics(symbol, delta_pos, cur_feei) #what metrics???
        self.trade_data[symbol]['o_longs'].append((last_x, new_price))
        self.trade_data[symbol]['count'] += 1
        self.trade_data[symbol]['amount'] += delta_pos

    def _handle_short_operations(self, symbol, new_pos, old_pos, new_price, last_x,delta_pos):
        """Обработка операций с шортами (открытие/добавление)"""
        abs_delta_pos = abs(delta_pos)
        feei = self.fee * new_price
        cur_feei = feei * abs_delta_pos
        
        if old_pos == 0:  # open_short
            self.trade_data[symbol]['mp'] = new_price
            self.open_fee[symbol] = cur_feei
        else:  # add_short
            old_price = self.trade_data[symbol]['mp']
            abs_old_pos = abs(old_pos)
            abs_new_pos = abs(new_pos)
            self.trade_data[symbol]['mp'] = (abs_old_pos * old_price + abs_delta_pos * new_price) / abs_new_pos
            self.open_fee[symbol] += cur_feei
        
        self._update_fee_metrics(symbol, abs_delta_pos, cur_feei)
        self.trade_data[symbol]['o_shorts'].append((last_x, new_price))
        self.trade_data[symbol]['count'] += 1
        self.trade_data[symbol]['amount'] += abs_delta_pos

    def _handle_short_closing(self, symbol, new_pos, old_pos, new_price, last_x,delta_pos):
        """Обработка закрытия шортовой позиции"""
        old_price = self.trade_data[symbol]['mp']
        delta = old_price - new_price
        old_pos_abs = abs(old_pos)
        feei = (self.fee * new_price) / 100
        cur_feei = feei * abs(delta_pos)
        
        if new_pos == 0:  # close_short completely
            self._close_short_completely(symbol, old_pos_abs, delta, new_price, cur_feei, last_x)
        elif new_pos > 0:  # close_short and open_long
            self._close_short_open_long(symbol, old_pos_abs, delta, new_price, delta_pos, cur_feei, last_x,new_pos)
        else:  # change_short (reduce short)
            self._reduce_short_position(symbol, new_pos, old_pos, delta, new_price, cur_feei, last_x)

    def _handle_long_closing(self, symbol, new_pos, old_pos, new_price, last_x,delta_pos):
        """Обработка закрытия лонговой позиции"""
        old_price = self.trade_data[symbol]['mp']
        delta = new_price - old_price
        abs_delta_pos = abs(delta_pos)
        feei = (self.fee * new_price) / 100
        cur_feei = feei * abs_delta_pos
        
        if new_pos == 0:  # close_long completely
            self._close_long_completely(symbol, old_pos, delta, new_price, cur_feei, last_x)
        elif new_pos < 0:  # close_long and open_short
            self._close_long_open_short(symbol, old_pos, delta, new_price, delta_pos, cur_feei, last_x,new_pos)
        else:  # change_long (reduce long)
            self._reduce_long_position(symbol, new_pos, old_pos, delta, new_price, cur_feei, last_x)

    def _close_short_completely(self, symbol, old_pos_abs, delta, new_price, cur_feei, last_x):
        """Полное закрытие шортовой позиции"""
        full_delta = delta * old_pos_abs
        reward = (((delta / new_price) * 100) - self.fee_one_p) * old_pos_abs
        
        self.trade_data[symbol]['total_wfees_per'] += reward
        self.trade_data[symbol]['fees'] += cur_feei
        
        last_equity = self.trade_data[symbol]['equity'][-1]
        last_equity_fee = self.trade_data[symbol]['equity_fee'][-1]
        self.trade_data[symbol]['equity'].append(last_equity + full_delta)
        self.trade_data[symbol]['equity_fee'].append(last_equity_fee + (full_delta - cur_feei - self.open_fee[symbol]))
        
        self.open_fee[symbol] = 0
        self.trade_data[symbol]['c_shorts'].append((last_x, new_price))

    def _close_short_open_long(self, symbol, old_pos_abs, delta, new_price, delta_pos, cur_feei, last_x,new_pos):
        """Закрытие шорта и открытие лонга"""
        full_delta = delta * old_pos_abs
        reward = ((delta / new_price) * 100 * old_pos_abs) - self.fee_one_p * delta_pos

        self.trade_data[symbol]['total_wfees_per'] += reward
        self.trade_data[symbol]['fees'] += cur_feei
        
        last_equity = self.trade_data[symbol]['equity'][-1]
        last_equity_fee = self.trade_data[symbol]['equity_fee'][-1]
        self.trade_data[symbol]['equity'].append(last_equity + full_delta)
        self.trade_data[symbol]['equity_fee'].append(last_equity_fee + (full_delta - cur_feei - self.open_fee[symbol]))
        
        self.trade_data[symbol]['mp'] = new_price
        self.open_fee[symbol] = 0
        self.trade_data[symbol]['c_shorts'].append((last_x, new_price))
        self.trade_data[symbol]['o_longs'].append((last_x, new_price))
        self.trade_data[symbol]['count'] += 1
        self.trade_data[symbol]['amount'] += new_pos

    def _reduce_short_position(self, symbol, new_pos, old_pos, delta, new_price, cur_feei, last_x):
        """Частичное закрытие шортовой позиции"""
        delta_short = abs(new_pos - old_pos)  # количество закрытых шортов
        partial_delta = delta * delta_short
        reward = (((delta / new_price) * 100) - self.fee_one_p) * delta_short
        
        self.trade_data[symbol]['total_wfees_per'] += reward
        self.trade_data[symbol]['fees'] += cur_feei
        
        last_equity = self.trade_data[symbol]['equity'][-1]
        last_equity_fee = self.trade_data[symbol]['equity_fee'][-1]
        
        # Пересчитываем open_fee для оставшейся позиции
        new_pos_abs = abs(new_pos)
        old_pos_abs = abs(old_pos)
        fee_ratio = new_pos_abs / old_pos_abs
        equity_fee_delta = partial_delta - cur_feei - self.open_fee[symbol] * (delta_short / old_pos_abs)
        
        self.trade_data[symbol]['equity'].append(last_equity + partial_delta)
        self.trade_data[symbol]['equity_fee'].append(last_equity_fee + equity_fee_delta)
        self.open_fee[symbol] = self.open_fee[symbol] * fee_ratio
        self.trade_data[symbol]['c_shorts'].append((last_x, new_price))

    def _close_long_completely(self, symbol, old_pos, delta, new_price, cur_feei, last_x):
        """Полное закрытие лонговой позиции"""
        full_delta = delta * old_pos
        reward = (((delta / new_price) * 100) - self.fee_one_p) * old_pos
        
        self.trade_data[symbol]['total_wfees_per'] += reward
        self.trade_data[symbol]['fees'] += cur_feei
        
        last_equity = self.trade_data[symbol]['equity'][-1]
        last_equity_fee = self.trade_data[symbol]['equity_fee'][-1]
        self.trade_data[symbol]['equity'].append(last_equity + full_delta)
        self.trade_data[symbol]['equity_fee'].append(last_equity_fee + (full_delta - cur_feei - self.open_fee[symbol]))
        
        self.open_fee[symbol] = 0
        self.trade_data[symbol]['c_longs'].append((last_x, new_price))

    def _close_long_open_short(self, symbol, old_pos, delta, new_price, delta_pos, cur_feei, last_x,new_pos):
        """Закрытие лонга и открытие шорта"""
        full_delta = delta * old_pos
        reward = ((delta / new_price) * 100 * old_pos) - self.fee_one_p * abs(delta_pos)
        
        self.trade_data[symbol]['total_wfees_per'] += reward
        self.trade_data[symbol]['fees'] += cur_feei
        
        last_equity = self.trade_data[symbol]['equity'][-1]
        last_equity_fee = self.trade_data[symbol]['equity_fee'][-1]
        self.trade_data[symbol]['equity'].append(last_equity + full_delta)
        self.trade_data[symbol]['equity_fee'].append(last_equity_fee + (full_delta - cur_feei - self.open_fee[symbol]))
        
        self.trade_data[symbol]['mp'] = new_price
        self.open_fee[symbol] = ((self.fee * new_price) / 100) * abs(delta_pos)
        self.trade_data[symbol]['c_longs'].append((last_x, new_price))
        self.trade_data[symbol]['o_shorts'].append((last_x, new_price))
        self.trade_data[symbol]['count'] += 1
        self.trade_data[symbol]['amount'] += abs(new_pos)

    def _reduce_long_position(self, symbol, new_pos, old_pos, delta, new_price, cur_feei, last_x):
        """Частичное закрытие лонговой позиции"""
        delta_long = old_pos - new_pos  # количество закрытых лонгов
        partial_delta = delta * delta_long
        reward = (((delta / new_price) * 100) - self.fee_one_p) * delta_long
        
        self.trade_data[symbol]['total_wfees_per'] += reward
        self.trade_data[symbol]['fees'] += cur_feei
        
        last_equity = self.trade_data[symbol]['equity'][-1]
        last_equity_fee = self.trade_data[symbol]['equity_fee'][-1]
        
        # Пересчитываем open_fee для оставшейся позиции
        fee_ratio = new_pos / old_pos
        equity_fee_delta = partial_delta - cur_feei - self.open_fee[symbol] * (delta_long / old_pos)
        
        self.trade_data[symbol]['equity'].append(last_equity + partial_delta)
        self.trade_data[symbol]['equity_fee'].append(last_equity_fee + equity_fee_delta)
        self.open_fee[symbol] = self.open_fee[symbol] * fee_ratio
        self.trade_data[symbol]['c_longs'].append((last_x, new_price))

    def _update_fee_metrics(self, symbol, delta_pos, cur_feei):
        """Обновление метрик связанных с комиссиями"""
        self.trade_data[symbol]['total_wfees_per'] -= self.fee_one_p * abs(delta_pos)
        self.trade_data[symbol]['fees'] += cur_feei

    # ==========================================================================

    @duration_time
    def check_fast_old(self):
        poss = self._check_position()
        self.ws.preprocessing(self.charts,poss)
        dfs = self.get_deep_copy_last_dfs()
        tf1 = self.timeframes[0]
        dates_df1 = dfs[tf1][self.symbols[0]]['ms'].to_list()
        for d in dates_df1:
            for tf in dfs:
                for s in dfs[tf]:
                    df = dfs[tf][s].copy()
                    filtered_df = df[df['ms'] <= d]
                    # Обновляем датафрейм в основном хранилище
                    self.ws.last_dfs[tf][s] = filtered_df
            need_pos = self.ws()
            last_prices = {s: self.ws.last_dfs[tf1][s].iloc[-1]['close'] for s in self.symbols}
            last_xs = {s: self.ws.last_dfs[tf1][s].iloc[-1]['x'] for s in self.symbols}
            if self.close_on_time:
                last_row = self.ws.last_dfs[tf1][self.symbols[0]].iloc[-1]
                time_close = self.close_map[last_row['weekday']]
                if last_row['ms'].hour >= time_close[0] and last_row['ms'].minute >= time_close[1]:
                    for s in need_pos:
                        need_pos[s] = 0
            self.work_need_pos(need_pos,last_prices,last_xs)

    @duration_time
    def check_fast(self):
        poss = self._check_position()
        self.ws.preprocessing(self.charts, poss)
        dfs = self.get_deep_copy_last_dfs()
        tf1 = self.timeframes[0]
        
        # Получаем все даты один раз
        dates_df1 = dfs[tf1][self.symbols[0]]['ms'].values  # numpy array вместо list
        
        # Предварительно индексируем данные для быстрой фильтрации
        indexed_dfs = {}
        for tf in dfs:
            indexed_dfs[tf] = {}
            for s in dfs[tf]:
                df = dfs[tf][s]
                # Создаем индекс по времени для быстрого поиска
                indexed_dfs[tf][s] = df.set_index('ms').sort_index()
        
        for d in dates_df1:
            # Быстрая фильтрация через .loc
            for tf in indexed_dfs:
                for s in indexed_dfs[tf]:
                    filtered_df = indexed_dfs[tf][s].loc[:d].reset_index()
                    self.ws.last_dfs[tf][s] = filtered_df
            
            need_pos = self.ws()
            
            # Оптимизируем получение последних цен
            last_prices = {}
            last_xs = {}
            for s in self.symbols:
                last_row = self.ws.last_dfs[tf1][s].iloc[-1]
                last_prices[s] = last_row['close']
                last_xs[s] = last_row['x']
            
            if self.close_on_time:
                last_row = self.ws.last_dfs[tf1][self.symbols[0]].iloc[-1]
                time_close = self.close_map[last_row['weekday']]
                if last_row['ms'].hour >= time_close[0] and last_row['ms'].minute >= time_close[1]:
                    for s in need_pos:
                        need_pos[s] = 0
            
            self.work_need_pos(need_pos, last_prices, last_xs)
            data = self.trade_data[self.symbols[0]]
            with open('logs/test_log.txt','a') as f:
                content = '\n'
                for k in ('total','count','amount','total_wfees_per','pos','mp','fees'):
                    content += k + ': ' + str(data[k]) + ' '
                f.write(content)

    @duration_time
    def check_window_fast(self, window_size=150):
        tf1 = self.timeframes[0]
        
        # Предварительная индексация данных
        indexed_charts = {}
        for tf in self.charts:
            indexed_charts[tf] = {}
            for s in self.charts[tf]:
                df = self.charts[tf][s].copy()
                df['timestamp'] = pd.to_datetime(df['ms'])
                indexed_charts[tf][s] = df.set_index('timestamp').sort_index()
        
        dates_df1 = indexed_charts[tf1][self.symbols[0]].index
        cache = {}  # Кэш для окон данных
        
        for i, d in enumerate(dates_df1):
            if i < window_size:
                continue
                
            poss = self._check_position()
            temp_dfs = {tf: {} for tf in self.timeframes}
            
            for tf in indexed_charts:
                for s in indexed_charts[tf]:
                    cache_key = (tf, s, d)
                    if cache_key in cache:
                        filtered_df = cache[cache_key]
                    else:
                        # Быстрый срез по индексу
                        filtered_df = indexed_charts[tf][s].loc[:d].tail(window_size).reset_index()
                        cache[cache_key] = filtered_df
                    
                    temp_dfs[tf][s] = filtered_df
            
            self.ws.preprocessing(temp_dfs, poss)
            need_pos = self.ws()
            last_prices = {s: self.ws.last_dfs[tf1][s].iloc[-1]['close'] for s in self.symbols}
            last_xs = {s: self.ws.last_dfs[tf1][s].iloc[-1]['x'] for s in self.symbols}
            if self.close_on_time:
                last_row = self.ws.last_dfs[tf1][self.symbols[0]].iloc[-1]
                time_close = self.close_map[last_row['weekday']]
                if last_row['ms'].hour >= time_close[0] and last_row['ms'].minute >= time_close[1]:
                    for s in need_pos:
                        need_pos[s] = 0
            self.work_need_pos(need_pos,last_prices,last_xs)



    @duration_time
    def check_window_old(self,window_size=150):
        tf1 = self.timeframes[0]
        dates_df1 = self.charts[tf1][self.symbols[0]]['ms'].to_list()
        for i,d in enumerate(dates_df1):
            if i < window_size:
                continue
            poss = self._check_position()
            temp_dfs = {tf: {} for tf in self.timeframes}
            for tf in self.charts:
                for s in self.charts[tf]:
                    df = self.charts[tf][s].copy()
                    filtered_df = df[df['ms'] <= d]
                    if len(filtered_df) > window_size:
                        filtered_df = filtered_df.iloc[-window_size:].copy()
                    # Обновляем датафрейм в основном хранилище
                    temp_dfs[tf][s] = filtered_df
            self.ws.preprocessing(temp_dfs,poss)
            need_pos = self.ws()
            last_prices = {s: self.ws.last_dfs[tf1][s].iloc[-1]['close'] for s in self.symbols}
            last_xs = {s: self.ws.last_dfs[tf1][s].iloc[-1]['x'] for s in self.symbols}
            if self.close_on_time:
                last_row = self.ws.last_dfs[tf1][self.symbols[0]].iloc[-1]
                time_close = self.close_map[last_row['weekday']]
                if last_row['ms'].hour >= time_close[0] and last_row['ms'].minute >= time_close[1]:
                    for s in need_pos:
                        need_pos[s] = 0
            self.work_need_pos(need_pos,last_prices,last_xs)

    @duration_time
    def check_child(self, window_size=150, timeframe='5min'):
        """
        Тестирование через младшие таймфреймы (child timeframe)
        """
        tf1 = self.timeframes[0]
        
        # Конвертируем основной таймфрейм в целевой используя готовую функцию
        big_timeframe_dfs = {}
        for s in self.symbols:
            df_main = self.charts[tf1][s].copy()
            big_timeframe_dfs[s] = convert_timeframe(df_main, timeframe)
        
        # Добавляем флаг закрытия если нужно
        if self.close_on_time:
            for s in self.symbols:
                df_big = big_timeframe_dfs[s]
                df_main = self.charts[tf1][s]
                df_big['close_2330'] = ((df_big['ms'].dt.hour == 23) & 
                                    (df_big['ms'].dt.minute > 25))
                df_main['close_2330'] = ((df_main['ms'].dt.hour == 23) & 
                                    (df_main['ms'].dt.minute > 25))
        
        # Проходим по всем барам большого таймфрейма
        period2x = 300
        big_dates = list(big_timeframe_dfs[self.symbols[0]]['ms'])
        
        for i in tqdm(range(period2x, len(big_dates))):
            current_big_time = big_dates[i]
            prev_big_time = big_dates[i-1]
            
            # Получаем окно данных для ВСЕХ таймфреймов
            temp_dfs_base = {tf: {} for tf in self.timeframes}
            
            for tf in self.timeframes:
                for s in self.symbols:
                    if tf == timeframe:
                        # Для целевого таймфрейма используем конвертированные данные
                        df_big = big_timeframe_dfs[s]
                        temp_dfs_base[tf][s] = df_big.iloc[i-period2x:i].copy()
                    else:
                        # Для остальных таймфреймов фильтруем данные до текущего времени
                        df_tf = self.charts[tf][s]
                        mask = df_tf['ms'] <= current_big_time
                        filtered_df = df_tf[mask].copy()
                        if len(filtered_df) > window_size:
                            filtered_df = filtered_df.iloc[-window_size:]
                        temp_dfs_base[tf][s] = filtered_df
            
            # Получаем child свечи для каждого символа
            child_candles_all = {}
            for s in self.symbols:
                df_main = self.charts[tf1][s]
                mask = (df_main['ms'] >= prev_big_time) & (df_main['ms'] < current_big_time)
                df_child = df_main[mask].copy()
                
                if len(df_child) > 0:
                    child_candles_all[s] = self._get_child_candles(df_child, temp_dfs_base[timeframe][s].iloc[-1])
                else:
                    child_candles_all[s] = []
            
            # Обрабатываем каждую child свечу
            max_child_candles = max(len(candles) for candles in child_candles_all.values()) if child_candles_all else 0
            
            for j in range(max_child_candles):
                # Создаем временные датафреймы с обновленной последней свечой
                temp_dfs = {tf: {} for tf in self.timeframes}
                
                # Копируем базовые данные
                for tf in self.timeframes:
                    for s in self.symbols:
                        temp_dfs[tf][s] = temp_dfs_base[tf][s].copy()
                
                # Обновляем последнюю свечу в целевом таймфрейме для всех символов
                for s in self.symbols:
                    child_candles = child_candles_all.get(s, [])
                    if j < len(child_candles) and timeframe in temp_dfs and s in temp_dfs[timeframe]:
                        if len(temp_dfs[timeframe][s]) > 0:
                            # Безопасное обновление через создание новой строки
                            child_candle = child_candles[j]
                            updated_row = temp_dfs[timeframe][s].iloc[-1:].copy()
                            
                            # Обновляем значения в копии строки
                            for col in child_candle.index:
                                if col in updated_row.columns:
                                    target_dtype = temp_dfs[timeframe][s][col].dtype
                                    updated_row[col] = self._safe_type_conversion(child_candle[col], target_dtype)
                            
                            # Заменяем последнюю строку обновленной версией
                            temp_dfs[timeframe][s] = pd.concat([
                                temp_dfs[timeframe][s].iloc[:-1], 
                                updated_row
                            ], ignore_index=True)
                
                # Получаем текущие позиции
                poss = self._check_position()
                
                # Препроцессинг и получение сигналов
                self.ws.preprocessing(temp_dfs, poss)
                need_pos = self.ws()
                
                # Получаем последние цены из child свечей
                last_prices = {}
                last_xs = {}
                for s in self.symbols:
                    child_candles = child_candles_all.get(s, [])
                    if j < len(child_candles):
                        last_row = child_candles[j]
                        last_prices[s] = last_row['close']
                        last_xs[s] = last_row['x']
                    elif child_candles:
                        last_candle = child_candles[-1]
                        last_prices[s] = last_candle['close']
                        last_xs[s] = last_candle['x']
                    else:
                        # Если нет child свечей, используем цену из большого таймфрейма
                        last_prices[s] = temp_dfs_base[timeframe][s].iloc[-1]['close']
                        last_xs[s] = temp_dfs_base[timeframe][s].iloc[-1]['x']
                
                # Принудительное закрытие по времени если нужно
                if self.close_on_time:
                    for s in self.symbols:
                        child_candles = child_candles_all.get(s, [])
                        if j < len(child_candles):
                            last_row = child_candles[j]
                            time_close = self.close_map[last_row['weekday']]
                            if (last_row['ms'].hour > time_close[0] or 
                                (last_row['ms'].hour == time_close[0] and last_row['ms'].minute >= time_close[1])):
                                need_pos[s] = 0
                        elif 'close_2330' in temp_dfs_base[timeframe][s].columns and temp_dfs_base[timeframe][s].iloc[-1]['close_2330']:
                            need_pos[s] = 0
                
                # Обрабатываем изменение позиций
                self.work_need_pos(need_pos, last_prices, last_xs)
                
                # Логирование для всех символов
                if self.need_debug:
                    for s in self.symbols:
                        data = self.trade_data[s]
                        with open(f'logs/child_test_log_{s}_{timeframe}.txt', 'a') as f:
                            content = f'\n{s}: '
                            for k in ('total', 'count', 'amount', 'total_wfees_per', 'pos', 'mp', 'fees'):
                                content += k + ': ' + str(data[k]) + ' '
                            f.write(content)

    def _safe_type_conversion(self, value, target_dtype):
        """
        Безопасное преобразование типа данных
        """
        try:
            if pd.isna(value):
                return value
            return target_dtype.type(value)
        except (ValueError, TypeError):
            try:
                # Альтернативное преобразование через numpy
                return np.array([value]).astype(target_dtype)[0]
            except:
                return value

    def _get_child_candles(self, df: pd.DataFrame, base_candle: pd.Series) -> list:
        """
        Генерирует child свечи с явным преобразованием типов данных
        """
        candles = []
        df = df.reset_index(drop=True)
        
        for i, row in df.iterrows():
            if i == 0:
                # Создаем новую свечу с правильными типами данных
                candle = base_candle.copy()
                
                # Явно преобразуем OHLCV данные к тем же типам, что в base_candle
                candle['open'] = self._convert_dtype(row['open'], base_candle['open'])
                candle['high'] = self._convert_dtype(row['high'], base_candle['high'])
                candle['low'] = self._convert_dtype(row['low'], base_candle['low'])
                candle['close'] = self._convert_dtype(row['close'], base_candle['close'])
                candle['volume'] = self._convert_dtype(row['volume'], base_candle['volume'])
                candle['ms'] = row['ms']
            else:
                # Обновляем только OHLCV последней свечи с преобразованием типов
                candle['close'] = self._convert_dtype(row['close'], base_candle['close'])
                candle['volume'] = self._convert_dtype(candle['volume'] + row['volume'], base_candle['volume'])
                candle['high'] = self._convert_dtype(max(candle['high'], row['high']), base_candle['high'])
                candle['low'] = self._convert_dtype(min(candle['low'], row['low']), base_candle['low'])
            
            # Пересчитываем производные поля
            candle['middle'] = self._convert_dtype((candle['high'] + candle['low']) / 2, base_candle.get('middle', candle['high']))
            candle['direction'] = 1 if candle['open'] < candle['close'] else -1
            
            candles.append(candle.copy())
        
        return candles

    def _convert_dtype(self, value, reference_value):
        """
        Преобразует значение к тому же типу, что и reference_value
        """
        if pd.isna(value) or pd.isna(reference_value):
            return value
        
        ref_dtype = type(reference_value)
        
        try:
            if ref_dtype == np.float32:
                return np.float32(value)
            elif ref_dtype == np.float64:
                return np.float64(value)
            elif ref_dtype == np.int32:
                return np.int32(value)
            elif ref_dtype == np.int64:
                return np.int64(value)
            else:
                return ref_dtype(value)
        except (ValueError, TypeError):
            # Если преобразование не удалось, возвращаем как есть
            return value
    

    # Какие-то проблемы
    @duration_time
    def check_child_lite(self, timeframe='5min'):
        """
        Упрощенный аналог check_strategy_v6 для класса
        """
        tf1 = self.timeframes[0]
        
        # Конвертируем в старший таймфрейм
        big_dfs = {}
        for s in self.symbols:
            df_main = self.charts[tf1][s].copy()
            big_dfs[s] = convert_timeframe(df_main, timeframe)
        
        # Проходим по всем барам старшего таймфрейма
        period2x = 300
        big_dates = list(big_dfs[self.symbols[0]]['ms'])
        
        for i in tqdm(range(period2x, len(big_dates))):
            current_time = big_dates[i]
            prev_time = big_dates[i-1]
            
            # Окно данных для ВСЕХ таймфреймов
            temp_dfs_base = {tf: {} for tf in self.timeframes}
            
            for tf in self.timeframes:
                for s in self.symbols:
                    if tf == timeframe:
                        # Для целевого таймфрейма используем конвертированные данные
                        temp_dfs_base[tf][s] = big_dfs[s].iloc[i-period2x:i].copy()
                    else:
                        # Для остальных таймфреймов используем оригинальные данные
                        df_tf = self.charts[tf][s]
                        mask = df_tf['ms'] <= current_time
                        temp_dfs_base[tf][s] = df_tf[mask].copy()
            
            # Получаем минутные свечи внутри этого бара
            child_candles_all = {}
            for s in self.symbols:
                df_main = self.charts[tf1][s]
                mask = (df_main['ms'] >= prev_time) & (df_main['ms'] < current_time)
                df_child = df_main[mask].copy()
                
                if len(df_child) > 0:
                    child_candles_all[s] = self._get_simple_child_candles(df_child, temp_dfs_base[timeframe][s].iloc[-1])
                else:
                    child_candles_all[s] = []
            
            # Обрабатываем каждую минутную свечу
            max_candles = max(len(candles) for candles in child_candles_all.values()) if child_candles_all else 0
            
            for j in range(max_candles):
                # Копируем базовые данные для всех таймфреймов
                temp_dfs = {tf: {} for tf in self.timeframes}
                for tf in self.timeframes:
                    for s in self.symbols:
                        temp_dfs[tf][s] = temp_dfs_base[tf][s].copy()
                
                # Обновляем последний бар только в ЦЕЛЕВОМ таймфрейме
                for s in self.symbols:
                    if j < len(child_candles_all[s]) and timeframe in temp_dfs and s in temp_dfs[timeframe]:
                        child_candle = child_candles_all[s][j]
                        df_target = temp_dfs[timeframe][s]
                        
                        if len(df_target) > 0:
                            # Создаем новую строку с правильными типами
                            new_row = df_target.iloc[-1:].copy()
                            for col in child_candle.index:
                                if col in new_row.columns:
                                    new_row[col] = self._safe_assign(child_candle[col], new_row[col].dtype)
                            
                            # Заменяем последнюю строку
                            temp_dfs[timeframe][s] = pd.concat([df_target.iloc[:-1], new_row], ignore_index=True)
                
                # Стандартная логика
                poss = self._check_position()
                self.ws.preprocessing(temp_dfs, poss)
                need_pos = self.ws()
                
                last_prices = {}
                last_xs = {}
                for s in self.symbols:
                    if j < len(child_candles_all[s]):
                        last_row = child_candles_all[s][j]
                        last_prices[s] = last_row['close']
                        last_xs[s] = last_row['x']
                    else:
                        last_prices[s] = temp_dfs_base[timeframe][s].iloc[-1]['close']
                        last_xs[s] = temp_dfs_base[timeframe][s].iloc[-1]['x']
                
                if self.close_on_time:
                    for s in self.symbols:
                        if j < len(child_candles_all[s]):
                            last_row = child_candles_all[s][j]
                            time_close = self.close_map[last_row['weekday']]
                            if (last_row['ms'].hour > time_close[0] or 
                                (last_row['ms'].hour == time_close[0] and last_row['ms'].minute >= time_close[1])):
                                need_pos[s] = 0
                
                self.work_need_pos(need_pos, last_prices, last_xs)

    def _get_simple_child_candles(self, df: pd.DataFrame, base_candle: pd.Series) -> list:
        """
        Простая генерация child свечей
        """
        candles = []
        df = df.reset_index(drop=True)
        
        for i, row in df.iterrows():
            if i == 0:
                candle = base_candle.copy()
                candle['open'] = row['open']
                candle['high'] = row['high'] 
                candle['low'] = row['low']
                candle['close'] = row['close']
                candle['volume'] = row['volume']
                candle['ms'] = row['ms']
            else:
                candle['close'] = row['close']
                candle['volume'] += row['volume']
                candle['high'] = max(candle['high'], row['high'])
                candle['low'] = min(candle['low'], row['low'])
            
            candle['middle'] = (candle['high'] + candle['low']) / 2
            candle['direction'] = 1 if candle['open'] < candle['close'] else -1
            
            candles.append(candle.copy())
        
        return candles

    def _safe_assign(self, value, target_dtype):
        """
        Простое безопасное присвоение с преобразованием типа
        """
        try:
            return target_dtype.type(value)
        except:
            return value
        
    def print_statistics(self, symbol):
        """Печать статистики по торгам"""
        if symbol not in self.symbols:
            print(f"Символ {symbol} не найден")
            return
            
        td = self.trade_data[symbol]
        
        print(f"\n=== СТАТИСТИКА ДЛЯ {symbol} ===")
        print(f"Прибыль ВТБ: {td['step_eq_vtb'][-1]}")
        print(f"Всего сделок: {td['amount']}")
        print(f"Общий PnL: {td['equity'][-1]:.2f}")
        print(f"Комиссия ВТБ: {td['amount']*2}")
        print(f"Комиссии: {td['fees']:.2f}")
        # print(f"PnL с комиссиями (%): {td['total_wfees_per']:.2f}%")
        
        # if td['equity']:
        #     final_equity = td['equity'][-1]
        #     final_equity_fee = td['equity_fee'][-1]
        #     print(f"Финальное эквити (без комиссий): {final_equity:.2f}")
        #     print(f"Финальное эквити (с комиссиями): {final_equity_fee:.2f}")
        
        # print(f"Открыто лонгов: {len(td['o_longs'])}")
        # print(f"Открыто шортов: {len(td['o_shorts'])}")
        # print(f"Закрыто лонгов: {len(td['c_longs'])}")
        # print(f"Закрыто шортов: {len(td['c_shorts'])}")
        # # Рассчитываем дополнительную статистику
        # if td['count'] > 0:
        #     avg_trade = td['total'] / td['count']
        #     print(f"Средний PnL на сделку: {avg_trade:.2f}")
    #draw funcs
    def plot_equity(self,symbol):
        plt.plot(self.trade_data[symbol]['equity'],color='r')
        plt.plot(self.trade_data[symbol]['equity_fee'],color='b')
        plt.show()