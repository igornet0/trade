import pandas as pd
import numpy as np
from talib.abstract import *

class Indicater:

    def calculate_technical_indicators(df: pd.DataFrame, SMAtimeperiod: int = 10, RCItimeperiod: int = 14):
        # Преобразуем столбцы DataFrame в numpy массивы
        open_price = df['open'].apply(lambda x: float(x)).values
        high = df['high'].apply(lambda x: float(x)).values
        low = df['low'].apply(lambda x: float(x)).values
        close = df['close'].apply(lambda x: float(x)).values
        volume = df['volume'].apply(lambda x: float(x)).values

        # Вычисляем различные технические индикаторы с помощью библиотеки TA-Lib
        # Например, можно использовать Moving Average, Relative Strength Index (RSI), MACD и другие

        # Примеры вычисления индикаторов
        df[f'SMA_{SMAtimeperiod}'] = SMA(close, timeperiod=SMAtimeperiod)  # Простое скользящее среднее
        df[f'RSI_{RCItimeperiod}'] = RSI(close, timeperiod=RCItimeperiod)  # Относительная сила индекса
        macd, signal, hist = MACD(close)  # MACD

        df['MACD'] = macd
        df['MACD_signal_line'] = signal

        return df

    def atr(df:pd.DataFrame, n:int = 14):
        df.insert(len(df.columns), f"atr_{n}", 0.0)

        high = df['max']
        low = df['min']
        close = df['close']
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1, skipna=False)

        atr = tr.rolling(n).mean()

        for i, item in enumerate(atr):
            df.at[i, f"atr_{n}"] = round(item, 3) if not np.isnan(atr[i]) else 0.0

        return df


    def trend(df:pd.DataFrame, window_size=10):
        """
        Определяет тренд на основе скользящего среднего
        :param data: pandas.DataFrame с ценами акций
        :param window_size: размер окна скользящего среднего
        
        """
        # Получаем цены закрытия
        close_prices = df['close']
        
        # Вычисляем скользящее среднее
        ma = close_prices.rolling(window=window_size).mean()
        
        # Определяем направление тренда
        trend = np.where(close_prices > ma, 1, -1)
        
        trend = pd.Series(trend, index=df.index)
        df.insert(len(df.columns), f"trend_{window_size}", 0.0)
        for i, item in enumerate(trend):
            df.at[i, f"trend_{window_size}"] = round(item, 2) if not np.isnan(item) else 0.0
        
        return df


    def stochastic_oscillator(df:pd.DataFrame, period=14):
        """
        Функция для расчета стохастического осциллятора.
        :param period: период расчета (по умолчанию 14).
        
        """
        highs = df['max']
        lows = df['min']
        closes = df['close']

        # Рассчитываем %K
        lowest_lows = lows.rolling(window=period).min()
        highest_highs = highs.rolling(window=period).max()
        k_values = 100 * (closes - lowest_lows) / (highest_highs - lowest_lows)

        # Рассчитываем %D
        d_values = k_values.rolling(window=3).mean()

        df.insert(len(df.columns), f"k_stoh_{period}", 0.0)
        df.insert(len(df.columns), f"d_stoh_{period}", 0.0)
        for i, item in enumerate(zip(k_values, d_values)):
            df.at[i, f"k_stoh_{period}"] = round(item[0], 2) if not np.isnan(item[0]) else 0.0
            df.at[i, f"d_stoh_{period}"] = round(item[1], 2) if not np.isnan(item[1]) else 0.0

        return df


    def support_resistance(df:pd.DataFrame, window_size:int=10, threshold:int=0.05):
        """
        Определяет уровни сопротивления и поддержки на основе скользящего среднего
        :param data: pandas.DataFrame с ценами акций
        :param window_size: размер окна скользящего среднего
        :param threshold: пороговое значение отклонения от скользящего среднего для определения уровня
        """
        # Получаем цены закрытия
        close_prices = df['close']
        
        # Вычисляем скользящее среднее
        ma = close_prices.rolling(window=window_size).mean()
        
        # Определяем уровни сопротивления и поддержки
        resistance = ma * (1 + threshold)
        support = ma * (1 - threshold)
        
        support_resistance = pd.DataFrame({'resistance': resistance, 'support': support}, index=df.index)
        df.insert(len(df.columns), f"resistance_{window_size}", 0.0)
        df.insert(len(df.columns), f"support_{window_size}", 0.0)
        for i, item in support_resistance.iterrows():
            df.at[i, f"resistance_{window_size}"] = round(item["resistance"], 2) if not np.isnan(item["resistance"]) else 0.0
            df.at[i, f"support_{window_size}"] = round(item["support"], 2) if not np.isnan(item["support"]) else 0.0

        return df

