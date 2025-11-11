import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from traders.TraderBase import TraderBase
from wss.WSBase import WSBase

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
        self.fee = fee
        self.charts = charts
        self.read_dfs()
        self.get_total_time_range()
        self.trade_data = {
            symbol : {
                'total':0,
                'count':0,
                'fees': 0,
                'total_wfees_per':0,
                'equity':[0],
                'equity_fee':[0],
                'pos':0,
                'mp':0,
                'o_longs':[],
                'o_shorts':[],
                'c_longs':[],
                'c_shorts':[]
            } for symbol in self.symbols
        }
        self.fee_one_p = (fee / 2) * 100
        self.open_fee = {symbol:0 for symbol in self.symbols}
        self.close_on_time = close_on_time
        self.close_map = close_map

    def read_dfs(self):
        for t in self.charts:
            for s in self.charts[t]:
                path_df = self.charts[t][s]
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

    def work_need_pos(self, need_pos, last_prices, last_xs):
        for s in self.symbols:
            if need_pos[s] != self.trade_data[s]['pos']:
                new_pos = need_pos[s]
                old_pos = self.trade_data[s]['pos']
                new_price = last_prices[s]
                feei = (self.fee * new_price) / 100
                old_price = self.trade_data[s]['mp']
                lx = last_xs[s]
                delta_pos = new_pos - old_pos
                cur_feei = feei * abs(delta_pos)
                abs_old_pos = abs(old_pos)
                abs_new_pos = abs(new_pos)
                last_equity = self.trade_data[s]['equity'][-1]
                last_equity_fee = self.trade_data[s]['equity_fee'][-1]
                
                # long operations
                if delta_pos > 0: # add
                    if old_pos > -1: # add_long or open_long
                        if old_pos == 0: # open_long
                            self.trade_data[s]['mp'] = new_price
                            self.open_fee[s] = cur_feei
                        else: # add_long
                            self.trade_data[s]['mp'] = (old_price * old_pos + new_price * delta_pos) / new_pos
                            self.open_fee[s] += cur_feei
                        self.trade_data[s]['total_wfees_per'] -= self.fee_one_p * delta_pos
                        self.trade_data[s]['fees'] += cur_feei
                        self.trade_data[s]['o_longs'].append((lx, new_price))
                        self.trade_data[s]['count'] += 1
                    else: # close_short (pos < 0) 
                        new_pos_abs = abs(new_pos)
                        old_pos_abs = abs(old_pos)
                        delta = old_price - new_price
                        
                        if new_pos == 0: # close_short completely
                            full_delta = delta * old_pos_abs
                            self.trade_data[s]['total'] += full_delta
                            reward = (((delta / new_price) * 100) - self.fee_one_p) * old_pos_abs
                            self.trade_data[s]['total_wfees_per'] += reward
                            self.trade_data[s]['fees'] += cur_feei
                            self.trade_data[s]['equity'].append(last_equity + full_delta)
                            self.trade_data[s]['equity_fee'].append(last_equity_fee + (full_delta - cur_feei - self.open_fee[s]))
                            self.open_fee[s] = 0
                            self.trade_data[s]['c_shorts'].append((lx, new_price))
                            
                        elif new_pos > 0: # close_short and open_long
                            full_delta = delta * old_pos_abs
                            reward = ((delta / new_price) * 100 * old_pos_abs) - self.fee_one_p * delta_pos
                            self.trade_data[s]['total'] += full_delta
                            self.trade_data[s]['total_wfees_per'] += reward
                            self.trade_data[s]['fees'] += cur_feei
                            self.trade_data[s]['equity'].append(last_equity + full_delta)
                            self.trade_data[s]['equity_fee'].append(last_equity_fee + (full_delta - cur_feei - self.open_fee[s]))
                            self.trade_data[s]['mp'] = new_price
                            self.open_fee[s] = feei * new_pos_abs
                            self.trade_data[s]['o_longs'].append((lx, new_price))
                            
                        else: # change_short (reduce short)
                            delta_short = abs(delta_pos)  # количество закрытых шортов
                            partial_delta = delta * delta_short
                            reward = (((delta / new_price) * 100) - self.fee_one_p) * delta_short
                            self.trade_data[s]['total'] += partial_delta
                            self.trade_data[s]['total_wfees_per'] += reward
                            self.trade_data[s]['fees'] += cur_feei
                            self.trade_data[s]['equity'].append(last_equity + partial_delta)
                            # Пересчитываем open_fee для оставшейся позиции
                            fee_ratio = new_pos_abs / old_pos_abs
                            equity_fee_delta = partial_delta - cur_feei - self.open_fee[s] * (delta_short / old_pos_abs)
                            self.trade_data[s]['equity_fee'].append(last_equity_fee + equity_fee_delta)
                            self.open_fee[s] = self.open_fee[s] * fee_ratio
                            self.trade_data[s]['c_shorts'].append((lx, new_price))
                
                # short operations  
                elif delta_pos < 0: # sub
                    if old_pos < 1: # add_short or open_short
                        if old_pos == 0: # open_short
                            self.trade_data[s]['mp'] = new_price
                            self.open_fee[s] = cur_feei
                        else: # add_short
                            self.trade_data[s]['mp'] = (abs_old_pos * old_price + abs(delta_pos) * new_price) / abs_new_pos
                            self.open_fee[s] += cur_feei
                        self.trade_data[s]['total_wfees_per'] -= self.fee_one_p * abs(delta_pos)
                        self.trade_data[s]['fees'] += cur_feei
                        self.trade_data[s]['o_shorts'].append((lx, new_price))
                        self.trade_data[s]['count'] += 1
                    else: # close_long (pos > 0)
                        new_pos_abs = abs(new_pos)
                        delta = new_price - old_price
                        
                        if new_pos == 0: # close_long completely
                            full_delta = delta * old_pos
                            reward = (((delta / new_price) * 100) - self.fee_one_p) * old_pos
                            self.trade_data[s]['total'] += full_delta
                            self.trade_data[s]['total_wfees_per'] += reward
                            self.trade_data[s]['fees'] += cur_feei
                            self.trade_data[s]['equity'].append(last_equity + full_delta)
                            self.trade_data[s]['equity_fee'].append(last_equity_fee + (full_delta - cur_feei - self.open_fee[s]))
                            self.open_fee[s] = 0
                            self.trade_data[s]['c_longs'].append((lx, new_price))
                            
                        elif new_pos < 0: # close_long and open_short
                            full_delta = delta * old_pos
                            reward = ((delta / new_price) * 100 * old_pos) - self.fee_one_p * abs(delta_pos)
                            self.trade_data[s]['total'] += full_delta
                            self.trade_data[s]['total_wfees_per'] += reward
                            self.trade_data[s]['fees'] += cur_feei
                            self.trade_data[s]['equity'].append(last_equity + full_delta)
                            self.trade_data[s]['equity_fee'].append(last_equity_fee + (full_delta - cur_feei - self.open_fee[s]))
                            self.trade_data[s]['mp'] = new_price
                            self.open_fee[s] = feei * new_pos_abs
                            self.trade_data[s]['o_shorts'].append((lx, new_price))
                            
                        else: # change_long (reduce long)
                            delta_long = old_pos - new_pos  # количество закрытых лонгов
                            partial_delta = delta * delta_long
                            reward = (((delta / new_price) * 100) - self.fee_one_p) * delta_long
                            self.trade_data[s]['total'] += partial_delta
                            self.trade_data[s]['total_wfees_per'] += reward
                            self.trade_data[s]['fees'] += cur_feei
                            self.trade_data[s]['equity'].append(last_equity + partial_delta)
                            # Пересчитываем open_fee для оставшейся позиции
                            fee_ratio = new_pos / old_pos
                            equity_fee_delta = partial_delta - cur_feei - self.open_fee[s] * (delta_long / old_pos)
                            self.trade_data[s]['equity_fee'].append(last_equity_fee + equity_fee_delta)
                            self.open_fee[s] = self.open_fee[s] * fee_ratio
                            self.trade_data[s]['c_longs'].append((lx, new_price))
                
                self.trade_data[s]['pos'] = new_pos


    def check_fast(self):
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
                if last_row['ms'].hour >= time_close[0] and last_row['ms'].minute >= time_close[0]:
                    for s in need_pos:
                        need_pos[s] = 0
            self.work_need_pos(need_pos,last_prices,last_xs)


            # for tf in self.ws.last_dfs:
            #     for s in self.ws.last_dfs[tf]:
            #         print(tf,s)
            #         self.ws.last_dfs[tf][s].info()



    
    def check_window(self,window_size=150):
        ...
    
    def check_child(self,timeframe='5min'):
        ...




    
    def visualize_results(self, symbol, figsize=(15, 12), save_path=None):
        """
        Визуализация результатов торгов для конкретного символа
        
        Args:
            symbol: символ для визуализации
            figsize: размер фигуры
            save_path: путь для сохранения графика (опционально)
        """
        if symbol not in self.symbols:
            print(f"Символ {symbol} не найден")
            return
            
        trade_data = self.trade_data[symbol]
        
        # Создаем фигуру с несколькими subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[3, 1, 1])
        fig.suptitle(f'Результаты торгов для {symbol}', fontsize=16, fontweight='bold')
        
        # 1. График цены с точками входа/выхода
        self._plot_price_with_trades(axes[0], symbol, trade_data)
        
        # 2. График позиции
        self._plot_position(axes[1], symbol, trade_data)
        
        # 3. График эквити
        self._plot_equity(axes[2], trade_data)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в {save_path}")
        
        plt.show()
    
    def _plot_price_with_trades(self, ax, symbol, trade_data):
        """График цены с точками входа и выхода"""
        # Получаем данные цены (используем первый таймфрейм)
        tf1 = self.timeframes[0]
        price_data = self.charts[tf1][symbol].copy()
        
        if not pd.api.types.is_datetime64_any_dtype(price_data['ms']):
            price_data['ms'] = pd.to_datetime(price_data['ms'], format='%Y-%m-%d %H:%M:%S')
        
        # Рисуем цену
        ax.plot(price_data['ms'], price_data['close'], label='Цена закрытия', linewidth=1, alpha=0.7, color='black')
        
        # Добавляем точки входа в лонги
        if trade_data['o_longs']:
            long_dates, long_prices = zip(*trade_data['o_longs'])
            ax.scatter(long_dates, long_prices, color='green', marker='^', s=50, 
                      label='Открытие лонг', alpha=0.7, zorder=5)
        
        # Добавляем точки входа в шорты
        if trade_data['o_shorts']:
            short_dates, short_prices = zip(*trade_data['o_shorts'])
            ax.scatter(short_dates, short_prices, color='red', marker='v', s=50, 
                      label='Открытие шорт', alpha=0.7, zorder=5)
        
        # Добавляем точки закрытия лонгов
        if trade_data['c_longs']:
            close_long_dates, close_long_prices = zip(*trade_data['c_longs'])
            ax.scatter(close_long_dates, close_long_prices, color='blue', marker='o', s=30, 
                      label='Закрытие лонг', alpha=0.7, zorder=5)
        
        # Добавляем точки закрытия шортов
        if trade_data['c_shorts']:
            close_short_dates, close_short_prices = zip(*trade_data['c_shorts'])
            ax.scatter(close_short_dates, close_short_prices, color='orange', marker='o', s=30, 
                      label='Закрытие шорт', alpha=0.7, zorder=5)
        
        ax.set_ylabel('Цена')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('График цены с точками входа/выхода')
        
        # Форматирование дат
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_position(self, ax, symbol, trade_data):
        """График позиции"""
        # Собираем все точки изменения позиции
        all_events = []
        
        # Добавляем открытия лонгов
        for date, price in trade_data['o_longs']:
            all_events.append((date, price, 'open_long'))
        
        # Добавляем открытия шортов
        for date, price in trade_data['o_shorts']:
            all_events.append((date, price, 'open_short'))
        
        # Добавляем закрытия лонгов
        for date, price in trade_data['c_longs']:
            all_events.append((date, price, 'close_long'))
        
        # Добавляем закрытия шортов
        for date, price in trade_data['c_shorts']:
            all_events.append((date, price, 'close_short'))
        
        # Сортируем по дате
        all_events.sort(key=lambda x: x[0])
        
        if all_events:
            # Восстанавливаем историю позиций
            dates = [event[0] for event in all_events]
            positions = [0]  # Начинаем с нулевой позиции
            
            for event in all_events:
                date, price, event_type = event
                current_pos = positions[-1]
                
                if event_type == 'open_long':
                    # Находим соответствующую сделку чтобы определить размер позиции
                    # Это упрощенная версия - в реальности нужно отслеживать размер позиции
                    positions.append(1)  # Предполагаем единичную позицию
                elif event_type == 'open_short':
                    positions.append(-1)  # Предполагаем единичную позицию
                elif event_type == 'close_long':
                    positions.append(0)
                elif event_type == 'close_short':
                    positions.append(0)
            
            # Убираем начальный 0 и строим ступенчатый график
            positions = positions[1:]
            ax.step(dates, positions, where='post', linewidth=2, color='purple')
            
            # Заливка для лонгов и шортов
            ax.fill_between(dates, positions, 0, where=np.array(positions) > 0, 
                           alpha=0.3, color='green', label='Лонг позиция')
            ax.fill_between(dates, positions, 0, where=np.array(positions) < 0, 
                           alpha=0.3, color='red', label='Шорт позиция')
        
        ax.set_ylabel('Позиция')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('График позиции')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    def _plot_equity(self, ax, trade_data):
        """График эквити"""
        equity = trade_data['equity']
        equity_fee = trade_data['equity_fee']
        
        # Создаем временную ось на основе количества точек
        x_points = range(len(equity))
        
        ax.plot(x_points, equity, label='Equity (без комиссий)', linewidth=2, color='blue')
        ax.plot(x_points, equity_fee, label='Equity (с комиссиями)', linewidth=2, color='red')
        
        ax.set_ylabel('Эквити')
        ax.set_xlabel('Номер сделки')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('График эквити')
        
        # Добавляем горизонтальную линию на уровне 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    def print_statistics(self, symbol):
        """Печать статистики по торгам"""
        if symbol not in self.symbols:
            print(f"Символ {symbol} не найден")
            return
            
        td = self.trade_data[symbol]
        
        print(f"\n=== СТАТИСТИКА ДЛЯ {symbol} ===")
        print(f"Всего сделок: {td['count']}")
        print(f"Общий PnL: {td['total']:.2f}")
        print(f"Комиссии: {td['fees']:.2f}")
        print(f"PnL с комиссиями (%): {td['total_wfees_per']:.2f}%")
        
        if td['equity']:
            final_equity = td['equity'][-1]
            final_equity_fee = td['equity_fee'][-1]
            print(f"Финальное эквити (без комиссий): {final_equity:.2f}")
            print(f"Финальное эквити (с комиссиями): {final_equity_fee:.2f}")
        
        print(f"Открыто лонгов: {len(td['o_longs'])}")
        print(f"Открыто шортов: {len(td['o_shorts'])}")
        print(f"Закрыто лонгов: {len(td['c_longs'])}")
        print(f"Закрыто шортов: {len(td['c_shorts'])}")
        
        # Рассчитываем дополнительную статистику
        if td['count'] > 0:
            avg_trade = td['total'] / td['count']
            print(f"Средний PnL на сделку: {avg_trade:.2f}")
    
    def compare_all_symbols(self, figsize=(12, 8)):
        """Сравнительная визуализация всех символов"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # График финального эквити
        symbols = []
        final_equities = []
        final_equities_fee = []
        
        for symbol in self.symbols:
            td = self.trade_data[symbol]
            if td['equity'] and td['equity_fee']:
                symbols.append(symbol)
                final_equities.append(td['equity'][-1])
                final_equities_fee.append(td['equity_fee'][-1])
        
        x = range(len(symbols))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], final_equities, width, label='Без комиссий', alpha=0.7)
        ax1.bar([i + width/2 for i in x], final_equities_fee, width, label='С комиссиями', alpha=0.7)
        
        ax1.set_ylabel('Финальное эквити')
        ax1.set_title('Сравнение финального эквити по символам')
        ax1.set_xticks(x)
        ax1.set_xticklabels(symbols, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График общего PnL
        totals = [self.trade_data[symbol]['total'] for symbol in symbols]
        ax2.bar(symbols, totals, alpha=0.7, color='orange')
        ax2.set_ylabel('Общий PnL')
        ax2.set_title('Общий PnL по символам')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Печатаем общую статистику
        print("\n=== ОБЩАЯ СТАТИСТИКА ===")
        total_pnl = sum(self.trade_data[symbol]['total'] for symbol in self.symbols)
        total_fees = sum(self.trade_data[symbol]['fees'] for symbol in self.symbols)
        total_trades = sum(self.trade_data[symbol]['count'] for symbol in self.symbols)
        
        print(f"Общий PnL всех символов: {total_pnl:.2f}")
        print(f"Общие комиссии: {total_fees:.2f}")
        print(f"Всего сделок: {total_trades}")