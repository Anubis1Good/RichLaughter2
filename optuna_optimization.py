import os
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import traceback
import psutil
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
# from strategies.test_strategies.check import check_strategy_v3
# from strategies.test_strategies.check import check_strategy_realistic_v1
# from Loader.BitgetLoader import bitget_loader
from utils.processing.add_vtb_fee_fut import get_func_vtb_fee
from utils.df_utils.convert_timeframe import convert_timeframe
phys_cores = psutil.cpu_count(logical=False) 

save_cores = 3
close_2330 = True
new_timeframe = None
# new_timeframe = '5min'
new_timeframe = '30min'
reverse_test = False
# reverse_test = True
# feature_optimization = 'total_abs_fee'
top_limit=600
# bottom_limit=150
bottom_limit=100
def check_strategy_v3():
    ...
# # Целевая функция для Optuna
def objective(trial, df, ws, param_options, fee=0.0002):
    """
    strategy_class: класс стратегии (например, PTA4_U3)
    param_options: кортеж с вариантами параметров ((5,10,15,...), ...)
    """
    df = df.copy()
    params = []
    for i, options in enumerate(param_options):
        param_name = f"param_{i}"
        
        # Категориальные параметры (строки)
        if isinstance(options[0], str):
            params.append(trial.suggest_categorical(param_name, options))
            continue
            
        # Числовые параметры
        unique_steps = {options[j+1]-options[j] for j in range(len(options)-1)}
        
        if len(unique_steps) == 1:  # Все шаги одинаковые
            step = unique_steps.pop()
            if isinstance(step, int):
                params.append(trial.suggest_int(param_name, min(options), max(options), step=step))
            else:
                params.append(trial.suggest_float(param_name, min(options), max(options), step=step))
        else:  # Неравномерные шаги
            if all(isinstance(x, int) for x in options):
                params.append(trial.suggest_int(param_name, min(options), max(options)))
            else:
                params.append(trial.suggest_float(param_name, min(options), max(options)))
    
    # 2. Создаём стратегию с текущими параметрами

    strategy = ws(
        "BTCUSDT", "1m", "usdt-futures",1,
        *params
    )
    df = strategy.preprocessing(df)
    # 3. Запускаем бэктест
    # result = check_strategy_realistic_v1(df, strategy,fee,close_2330)[0]
    result = check_strategy_v3(df, strategy,fee,close_2330)[0]
    if not bottom_limit <= result['count'] <= top_limit:
        raise optuna.TrialPruned()
    
    # 4. Возвращаем метрику (например, Sharpe Ratio)
    return result['total_abs_fee']

def get_optimization_results_table(study, df, strategy_class, param_options, need_plot=False, ticker='MXI',fee=0.0002):
    results = []
    name_bot = str(strategy_class.__name__)
    images_folder = os.path.join(main_folder, name_bot, ticker, 'Images')
    file_folder = os.path.join(main_folder, name_bot, ticker)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_top = 25 if len(completed_trials) > 25 else len(completed_trials)

    top_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)[:n_top]
    
    if not top_trials:
        return pd.DataFrame()
    
    for trial in top_trials:

        # Собираем параметры в правильном порядке
        params = []
        param_values = []
        for i, options in enumerate(param_options):
            param_name = f"param_{i}"
            param_value = trial.params[param_name]
            if isinstance(param_value,float):
                param_value = round(param_value,2)
            params.append(param_value)
            param_values.append(str(param_value))  # Для имени файла

        # Создаем стратегию
        strategy = strategy_class("BTCUSDT","1m", "usdt-futures", 1,*params)
        processed_df = strategy.preprocessing(df.copy())
        
        # Запускаем бэктест
        # trades, eq, eq_f,_,_,_ = check_strategy_realistic_v1(processed_df, strategy,fee,close_2330)
        trades, eq, eq_f = check_strategy_v3(processed_df, strategy,fee,close_2330)
        
        # Формируем имя файла
        name_doc = f"{ticker}_{name_bot}"
        name_file = f"{name_doc}_{'_'.join(param_values)}"
        if name_bot.startswith('Rev'):
            params_tuple = f"(get_rws({name_bot[3:]}),({','.join(param_values)},)),"
        else:
            params_tuple = f"({name_bot},({','.join(param_values)},)),"
        vtb_twf_func = get_func_vtb_fee(ticker.split('_')[0])
        # Добавляем результаты
        result_row = {
            # "Trial": trial.number,
            "name": name_file,
            "ws":params_tuple,
            "vtb": vtb_twf_func(trades["total"],trades["count"])
            # "Total": trades["total"],
            # "TradesCount": trades["count"],
            # "TotalFeePer": trades["total_fee_per"],  
        }
        result_row = result_row | trades
        # Добавляем параметры в результат
        result_row['origin'] = ticker
        for i, param_value in enumerate(params):
            result_row[f"Param_{i}"] = param_value
        results.append(result_row)

        # Сохраняем график если нужно
        if need_plot:
            full_name_img = os.path.join(images_folder, f"{name_file}.png")
            plt.figure(figsize=(12, 6))
            plt.plot(eq, color='red', label='Equity')
            plt.plot(eq_f, color='blue', label='Equity with Fees')
            plt.title(f"{name_bot} (Trial {trial.number})")
            plt.legend()
            plt.savefig(full_name_img, bbox_inches='tight')
            plt.close()
    
    # Сортируем результаты
    df_results = pd.DataFrame(results).sort_values('total_abs_fee', ascending=False)
    df_results = df_results.drop_duplicates(subset=['total_abs_fee'])
    df_results = df_results.reset_index(drop=True)
    full_name_doc = os.path.join(file_folder, name_doc + '.xlsx')
    with pd.ExcelWriter(full_name_doc, engine='xlsxwriter') as writer:  
        df_results.to_excel(writer, sheet_name='total')
        worksheet = writer.sheets['total']
        for i, col in enumerate(df_results.columns,start=1):
            width = max(df_results[col].apply(lambda x: len(str(x))).max(), len(col))
            worksheet.set_column(i, i, width)
    return df_results

def optimization_optuna(raw_file,strategy_config,n_trials=100,n_jobs=1,need_plot=False,fee=0.0002):
    sep = '\\' if '\\' in raw_file else '/'
    variant_name = raw_file.split(sep)[-1]
    variant_name = variant_name.split('_')
    variant_name = variant_name[0]+'_'+variant_name[1]
    if raw_file.endswith('.parquet'):
        df = pd.read_parquet(raw_file)
    else:
        df = pd.read_csv(raw_file)
    if new_timeframe:
        df = convert_timeframe(df,new_timeframe)
    study = optuna.create_study(direction="maximize")
    # # Оптимизируем (n_trials=100, можно увеличить)
    study.optimize(lambda trial: objective(trial, df,*strategy_config,fee), n_trials=n_trials,n_jobs=n_jobs)

    res = get_optimization_results_table(study,df,strategy_config[0],strategy_config[1],need_plot,variant_name,fee)
    return res

main_folder = 'TestNewResults/Optuna'
if not os.path.exists(main_folder):
    os.makedirs(main_folder)
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.ERROR)


def process_file(raw_file, part, n_trials, n_jobs, need_plot, min_fee):
    """Обработка одного файла в отдельном процессе"""
    try:
        print(f"\nProcessing {os.path.basename(raw_file)}")
        optimization_optuna(raw_file, part, n_trials, n_jobs, need_plot, min_fee)
        return True
    except Exception as err:
        traceback.print_exc()
        print(f"\nError processing {os.path.basename(raw_file)}: {str(err)}")
        return False

def process_group(part, test_folder, n_trials, n_jobs, need_plot, min_fee):
    """Параллельная обработка группы файлов"""
    print(f"\nStarting processing for {part[0]}")
    
    # Получаем список файлов для обработки
    try:
        files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) 
                if os.path.isfile(os.path.join(test_folder, f))]
    except Exception as e:
        print(f"Error reading directory {test_folder}: {str(e)}")
        return 0

    if not files:
        print("No files found in directory")
        return 0

    # Настраиваем количество процессов
    num_processes = min(max(1, phys_cores - save_cores), len(files))
    print(f"Using {num_processes} processes for {len(files)} files")

    # Создаем worker функцию с фиксированными параметрами
    worker = partial(process_file,
                    part=part,
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    need_plot=need_plot,
                    min_fee=min_fee)

    # Обрабатываем файлы параллельно
    success_count = 0
    with Pool(processes=num_processes) as pool:
        try:
            # Используем imap_unordered для более эффективной работы
            for result in pool.imap_unordered(worker, files):
            # for result in tqdm(pool.imap_unordered(worker, files),
            #                 total=len(files),
            #                 desc=f"Processing {part[0]}",
            #                 unit="file"):
                if result:
                    success_count += 1
        except Exception as e:
            print(f"Error in multiprocessing: {str(e)}")

    print(f"\nCompleted {success_count}/{len(files)} files successfully for {part[0]}")
    return success_count

if __name__ == '__main__':


    # from Optimiztion.optimizations_groups.optuna_exp_groups import group
    # from Optimiztion.optimizations_groups.optuna_groups import group

    # group = [(get_rws(x[0]),x[1]) for x in group]
    # test_folder = 'DataForTests\DataFromMOEX'
    # test_folder = 'DataForTests\DataFromMoexFast'
    # test_folder = 'DataForTests\DataFromMoexFastStock'
    test_folder = 'DataForTests\DataFromMoexForStepTests'
    # test_folder = 'DataForTests\DataFromMoexTemp'
    # test_folder = 'DataForTests\DataFromBitget'
    test_folder = 'DataForTests\DataFromBybit'
    min_fee = 0.001
    # min_fee: float = 0.0002
    # min_fee: float = 0.0004
    # min_fee: float = 0.00002
    # min_fee: float = 0.0004
    need_plot=True
    # n_trials = 100
    n_trials = 200
    # n_trials = 500
    n_jobs = 1


    # for part in group:
    #     process_group(part, test_folder, n_trials, n_jobs, need_plot, min_fee)
    