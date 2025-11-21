import os
import pandas as pd
import optuna
import traceback
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt

from traders.TestTrader.TestTrader import TestTrader

phys_cores = mp.cpu_count()
save_cores = 2

class TestTraderOptimizer:
    def __init__(self, main_folder='test_results\optuna_optimization', top_limit=600, bottom_limit=100):
        self.main_folder = main_folder
        self.top_limit = top_limit
        self.bottom_limit = bottom_limit
        
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)
        optuna.logging.disable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    def objective(self, trial, config):
        """
        trial: optuna trial
        config: конфигурация для TestTrader
        """
        # 1. Подбираем параметры стратегии
        ws_params = {}
        for param_name, options in config['ws_params_options'].items():
            # Сортируем options для корректного вычисления шага
            sorted_options = sorted(options)
            
            if isinstance(sorted_options[0], str):
                ws_params[param_name] = trial.suggest_categorical(param_name, sorted_options)
            elif isinstance(sorted_options[0], int):
                # Проверяем, равномерный ли шаг
                if len(sorted_options) > 1:
                    steps = [sorted_options[i+1] - sorted_options[i] for i in range(len(sorted_options)-1)]
                    unique_steps = set(steps)
                    
                    if len(unique_steps) == 1:  # Все шаги одинаковые
                        step = abs(unique_steps.pop())  # Берем абсолютное значение
                        ws_params[param_name] = trial.suggest_int(
                            param_name, 
                            min(sorted_options), 
                            max(sorted_options), 
                            step=step
                        )
                    else:  # Неравномерные шаги
                        ws_params[param_name] = trial.suggest_int(
                            param_name, 
                            min(sorted_options), 
                            max(sorted_options)
                        )
                else:  # Только одно значение
                    ws_params[param_name] = sorted_options[0]
            else:  # float
                # Проверяем, равномерный ли шаг
                if len(sorted_options) > 1:
                    steps = [sorted_options[i+1] - sorted_options[i] for i in range(len(sorted_options)-1)]
                    unique_steps = set(steps)
                    
                    if len(unique_steps) == 1:  # Все шаги одинаковые
                        step = abs(unique_steps.pop())  # Берем абсолютное значение
                        ws_params[param_name] = trial.suggest_float(
                            param_name, 
                            min(sorted_options), 
                            max(sorted_options), 
                            step=step
                        )
                    else:  # Неравномерные шаги
                        ws_params[param_name] = trial.suggest_float(
                            param_name, 
                            min(sorted_options), 
                            max(sorted_options)
                        )
                else:  # Только одно значение
                    ws_params[param_name] = sorted_options[0]
        
        # Остальной код без изменений...
        try:
            trader = TestTrader(
                symbols=config['symbols'],
                timeframes=config['timeframes'],
                quantity=config['quantity'],
                ws=(config['ws_class'], ws_params),
                charts=config['charts'],
                fee=config.get('fee', 0.0002),
                close_on_time=config.get('close_on_time', False),
                need_debug=False
            )
            trader.check_fast()
            
            total_trades = sum([trader.trade_data[s]['amount'] for s in trader.trade_data])
            
            if not (self.bottom_limit <= total_trades <= self.top_limit):
                raise optuna.TrialPruned()
            
            total_vtb_profit = sum([trader.trade_data[s]['step_eq_vtb'][-1] for s in trader.trade_data])
            
            return total_vtb_profit
            
        except Exception as e:
            print(f"Error in trial: {str(e)}")
            raise optuna.TrialPruned()
    
    def optimize_configuration(self, config, n_trials=100, n_jobs=1):
        """
        Оптимизация одной конфигурации
        """
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler()
        )
        
        objective_func = lambda trial: self.objective(trial, config)
        
        study.optimize(objective_func, n_trials=n_trials, n_jobs=n_jobs)
        
        # Сохраняем результаты
        self.save_optimization_results(study, config)
        
        return study
    
    def save_optimization_results(self, study, config):
        """Сохранение результатов оптимизации"""
        strategy_name = config['ws_class'].__name__
        symbol_name = "_".join(config['symbols'])
        
        # Папки для результатов
        strategy_folder = os.path.join(self.main_folder, strategy_name, symbol_name)
        images_folder = os.path.join(strategy_folder, "Images")
        os.makedirs(images_folder, exist_ok=True)
        
        # Топ trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)[:25]
        
        results = []
        for trial in top_trials:
            # Собираем параметры
            params = {name: trial.params[name] for name in config['ws_params_options'].keys()}
            
            # Создаем стратегию с лучшими параметрами для детального анализа
            trader = TestTrader(
                symbols=config['symbols'],
                timeframes=config['timeframes'],
                quantity=config['quantity'],
                ws=(config['ws_class'], params),
                charts=config['charts'],
                fee=config.get('fee', 0.0002),
                close_on_time=config.get('close_on_time', False),
                need_debug=False
            )
            
            # Запускаем тестирование
            trader.check_fast()
            
            # Собираем детальную статистику
            detailed_stats = {}
            total_trades = 0
            
            for symbol in config['symbols']:
                td = trader.trade_data[symbol]
                symbol_trades = td['amount']
                total_trades += symbol_trades
                
                detailed_stats[f"{symbol}_vtb_profit"] = td['step_eq_vtb'][-1]
                detailed_stats[f"{symbol}_total_trades"] = symbol_trades
                detailed_stats[f"{symbol}_total_pnl"] = td['equity'][-1]
                detailed_stats[f"{symbol}_fees"] = td['fees']
                detailed_stats[f"{symbol}_total_wfees_per"] = td['total_wfees_per']
            
            # Формируем результат
            result_row = {
                "trial_number": trial.number,
                "total_vtb_profit": trial.value,
                "total_trades": total_trades,
                "strategy": f"({strategy_name}, {params})",
                "symbols": ", ".join(config['symbols']),
            }
            result_row.update(params)
            result_row.update(detailed_stats)
            
            results.append(result_row)
            
            # Сохраняем график equity curve (для первого символа)
            if config['symbols']:
                main_symbol = config['symbols'][0]
                equity_data = trader.trade_data[main_symbol]['equity_fee']
                if len(equity_data) > 1:
                    plt.figure(figsize=(12, 6))
                    plt.plot(equity_data)
                    plt.title(f"{strategy_name} - Trial {trial.number} - Trades: {total_trades} - Profit: {trial.value:.2f}")
                    plt.savefig(os.path.join(images_folder, f"trial_{trial.number}.png"))
                    plt.close()
        
        # Сохраняем в Excel
        if results:
            df_results = pd.DataFrame(results)
            excel_path = os.path.join(strategy_folder, f"results_{symbol_name}.xlsx")
            
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                df_results.to_excel(writer, sheet_name='Results', index=False)
                worksheet = writer.sheets['Results']
                for i, col in enumerate(df_results.columns):
                    width = max(df_results[col].astype(str).apply(len).max(), len(col))
                    worksheet.set_column(i, i, width)
            
            print(f"Saved results to {excel_path}")
        
        return results

def process_single_config(config, n_trials=100, n_jobs=1, top_limit=600, bottom_limit=100):
    """Обработка одной конфигурации в отдельном процессе"""
    try:
        optimizer = TestTraderOptimizer(top_limit=top_limit, bottom_limit=bottom_limit)
        
        strategy_name = config['ws_class'].__name__
        symbol_name = "_".join(config['symbols'])
        print(f"Optimizing {strategy_name} for {symbol_name} (trades limit: {bottom_limit}-{top_limit})")
        
        study = optimizer.optimize_configuration(
            config=config,
            n_trials=n_trials,
            n_jobs=n_jobs
        )
        
        # Статистика по прогонам
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        print(f"Completed {strategy_name} for {symbol_name}. "
              f"Best value: {study.best_value:.2f}, "
              f"Completed: {len(completed_trials)}, "
              f"Pruned: {len(pruned_trials)}")
        
        return study
        
    except Exception as e:
        print(f"Error optimizing {config['ws_class'].__name__} for {config['symbols']}: {str(e)}")
        traceback.print_exc()
        return None

def main():
    from work_inits.optuna_configs import optimization_configs
    # Настройки лимитов
    top_limit = 1000
    bottom_limit = 100
    # Пример конфигураций для оптимизации
    
    n_trials = 50
    n_jobs = 3
    
    # Параллельная обработка конфигураций
    num_processes = min(phys_cores - save_cores, len(optimization_configs))
    
    print(f"Starting optimization of {len(optimization_configs)} configurations")
    print(f"Trade limits: {bottom_limit}-{top_limit} trades")
    print(f"Using {num_processes} processes")
    
    worker = partial(process_single_config, 
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    top_limit=top_limit,
                    bottom_limit=bottom_limit)
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(worker, optimization_configs),
            total=len(optimization_configs),
            desc="Optimizing configurations"
        ))
    
    # Анализ результатов
    successful_studies = [s for s in results if s is not None]
    total_completed = sum(len([t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]) 
                         for s in successful_studies)
    total_pruned = sum(len([t for t in s.trials if t.state == optuna.trial.TrialState.PRUNED]) 
                      for s in successful_studies)
    
    print(f"\nOptimization completed!")
    print(f"Successful: {len(successful_studies)}/{len(optimization_configs)} configurations")
    print(f"Total trials: Completed: {total_completed}, Pruned: {total_pruned}")

# Альтернативный вариант - оптимизация одной конфигурации
def optimize_single_config(config, n_trials=100, n_jobs=1, top_limit=600, bottom_limit=100):
    """Оптимизация одной конфигурации без multiprocessing"""
    optimizer = TestTraderOptimizer(top_limit=top_limit, bottom_limit=bottom_limit)
    return optimizer.optimize_configuration(config, n_trials=n_trials, n_jobs=n_jobs)

if __name__ == '__main__':
    # Можно запустить так для одной конфигурации
    # config = {...}  # ваша конфигурация
    # study = optimize_single_config(config, n_trials=100, top_limit=600, bottom_limit=100)
    
    # Или так для нескольких конфигураций
    main()