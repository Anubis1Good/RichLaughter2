import os
import pandas as pd
from tqdm import tqdm

main_folder = 'test_results\optuna_optimization'
ignore_folders = ('archive',)
amount_var = 1
inner_folders = os.listdir(main_folder)
df_total = pd.DataFrame()
for inner_folder in tqdm(inner_folders):
    inner_folder_path = os.path.join(main_folder,inner_folder)
    if inner_folder in ignore_folders or not os.path.isdir(inner_folder_path):
        continue
    variant_folders = os.listdir(inner_folder_path)
    for variant_folder in variant_folders:
        variant_folder_path = os.path.join(inner_folder_path,variant_folder)
        if not os.path.isdir(variant_folder_path):
            continue
        files = os.listdir(variant_folder_path)
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(variant_folder_path,file)
                df_file = pd.read_excel(file_path,'Results')
                if not df_file.empty:
                    if df_total.empty:
                        df_total = df_file.head(amount_var)
                    else:
                        df_total = pd.concat([df_total,df_file.head(amount_var)])
df_total = df_total.sort_values('total_vtb_profit',ascending=False)
df_total = df_total.reset_index(drop=True)
if 'Unnamed: 0' in df_total.columns:
    df_total = df_total.drop('Unnamed: 0',axis=1)
full_name_doc = os.path.join(main_folder, 'TotalOptuna' + '.xlsx')
with pd.ExcelWriter(full_name_doc, engine='xlsxwriter') as writer:  
    df_total.to_excel(writer, sheet_name='total')
    worksheet = writer.sheets['total']
    workbook = writer.book
    for i, col in enumerate(df_total.columns,start=1):
        width = max(df_total[col].apply(lambda x: len(str(x))).max(), len(col))
        worksheet.set_column(i, i, width)
        worksheet.conditional_format(1, i, len(df_total), i, {
            'type': 'cell',
            'criteria': 'less than',
            'value': 0,
            'format': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        })
        worksheet.conditional_format(1, i, len(df_total), i, {
            'type': '3_color_scale',
            'min_color': '#DA9694',
            'mid_color': '#FFFFFF',
            'max_color': '#00B0F0'
        })
