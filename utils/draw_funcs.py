import matplotlib.pyplot as plt
import pandas as pd

def draw_hb_chart_fast(df):
    # Разделяем данные по направлениям
    longs = df[df['direction'] == 1]
    shorts = df[df['direction'] != 1]
    
    # Рисуем все линии за один вызов для каждого направления
    plt.vlines(longs.index, longs['low'], longs['high'], colors='#b7ea00')
    plt.vlines(shorts.index, shorts['low'], shorts['high'], colors='#ff0013')

def draw_hbwv_chart(df):
    # Разделяем данные по направлениям
    longs = df[df['direction'] == 1]
    shorts = df[df['direction'] != 1]
    # Рисуем все линии за один вызов для каждого направления
    plt.subplot(2,1,1)
    plt.vlines(longs.index, longs['low'], longs['high'], colors='#b7ea00')
    plt.vlines(shorts.index, shorts['low'], shorts['high'], colors='#ff0013')
    plt.subplot(2,1,2)
    plt.vlines(longs.index, 0, longs['volume'], colors='#b7ea00')
    plt.vlines(shorts.index, 0, shorts['volume'], colors='#ff0013')

def draw_chart_channel(df,top='max_hb',bottom='min_hb',median='avarege'):
    plt.plot(df[top])
    plt.plot(df[bottom])
    plt.plot(df[median])

def draw_bollinger(df,clr='blue'):
    plt.plot(df['bbu'],color=clr)
    plt.plot(df['bbd'],color=clr)
    plt.plot(df['sma'],color=clr)



def draw_dynamics(df,clr='red'):
    df.apply(lambda row: plt.text(row.name,row['low'],round(row['dynamics_ma'],1), fontsize=10,rotation='vertical'),axis=1)

def draw_rails(df,clr='blue'):
    df.apply(lambda row: plt.vlines(row.name,row['low'],row['high'],colors=clr) if row['rails'] else None,axis=1)

def draw_fractals_williams(df):
    df.apply(lambda row: plt.scatter(row.name,row['high'],color='#d64040' ) if row['fractal_up'] else 1,axis=1)
    df.apply(lambda row: plt.scatter(row.name,row['low'],color='#74992b' ) if row['fractal_down'] else 1,axis=1)

def draw_rsi(df):
    plt.plot(df['rsi'])
    plt.axhline(70, color='gray', linestyle='--', label='Перекупленность (70)')
    plt.axhline(30, color='gray', linestyle='--', label='Перепроданность (30)')

def draw_stochastic(df):
    plt.plot(df['%K'],color='blue')
    plt.plot(df['%D'],color='green')