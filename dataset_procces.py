import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time

def dtimer(func):

    def start(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        timer = round(end - start, 3)
        print(f"{timer=}")

    return start


def printd(d: dict):
    for key, item in d.items():
        print(f"{key}: ", end="")
        print(*item, sep="\n")


def check_df(df: pd.DataFrame):
    int_end = df["time_int"].max()
    d = {}
    for i, item in df.iterrows():
        if len(d) != 2:
            d[len(d)] = item

        elif len(d) == 2:
            values_time = list(map(lambda x: x["time_int"], d.values()))

            if (values_time[0] != int_end or values_time[1] != 1) and (abs(values_time[1] - values_time[0]) != 1):
                printd(d)
                print("NOT CLEAR !")
                return
            
            d = {0: d[1], 1: item}
    print("CLEAR !")


def convert_value(value: str) -> float:
    if value == "x":
        return value
    
    if "K" in value:
        value = value.replace("K", "")
        value = float(value) / 10**3
    
    elif "M" in value:
        value = value.replace("M", "")
        value = float(value) 
    
    elif "B" in value:
        value = value.replace("B", "")
        value = float(value) * 10**3
    
    return round(float(value), 2)


def check_start_end(l_se:list, d:datetime) -> bool:
    for datetime in l_se:
        if datetime.time() == d.time():
            return True
    return False


def check_time_minute(d: datetime, timetravel:str = "5m") -> datetime:
    func_temp = lambda x: x.time().minute % int(timetravel.replace(timetravel[-1], ""))

    return d.replace(minute=int(input(f"{d=}\nminute = "))) if func_temp(d) != 0 else d
    #return d.replace(minute=25) if func_temp(d) != 0 else d


def confert_datetime_to_int_item(item:datetime, timetravel_dict:dict, timetravel:str = "5m") -> None:
        try:
            time_start = timetravel_dict["start"]
            timetravel_str = timetravel

            time_s = item.replace(hour=time_start.hour, minute=time_start.minute)
            n = int(timetravel_str.replace(timetravel_str[-1], ""))
            delta = int(abs(item - time_s).total_seconds() // (n*{"m": 60, "H": 60**2, "D":34*60**2}[timetravel_str[-1]]))

            return {"time_int": delta + 1, "day_int": item.isoweekday()}
        except TypeError:
            return None
        

def procces(df:pd.DataFrame, timetravel:str = "5m") -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df['datetime'])
    df["datetime"] = df["datetime"].apply(check_time_minute, args=(timetravel, ))

    df.drop(df.loc[df["datetime"].dt.hour > 18].index, inplace=True)

    df = df.sort_values('datetime', ignore_index=True)
    df = df.drop_duplicates(subset='datetime', ignore_index=True)
    return df


def get_atr(file_csv:str, n:int = 14):
    df = pd.read_csv(file_csv, index_col="Unnamed: 0")
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

    df.to_csv(file_csv)


def get_trend(file_csv:str, window_size=10):
    """
    Определяет тренд на основе скользящего среднего
    :param data: pandas.DataFrame с ценами акций
    :param window_size: размер окна скользящего среднего
    
    """
    # Получаем цены закрытия
    df = pd.read_csv(file_csv, index_col="Unnamed: 0")
    close_prices = df['close']
    
    # Вычисляем скользящее среднее
    ma = close_prices.rolling(window=window_size).mean()
    
    # Определяем направление тренда
    trend = np.where(close_prices > ma, 1, -1)
    
    trend = pd.Series(trend, index=df.index)
    df.insert(len(df.columns), f"trend_{window_size}", 0.0)
    for i, item in enumerate(trend):
        df.at[i, f"trend_{window_size}"] = round(item, 2) if not np.isnan(item) else 0.0
    
    df.to_csv(file_csv)


def get_stochastic_oscillator(file_csv:str, period=14):
    """
    Функция для расчета стохастического осциллятора.

    :param high_prices: массив с максимальными ценами.
    :param low_prices: массив с минимальными ценами.
    :param close_prices: массив с ценами закрытия.
    :param period: период расчета (по умолчанию 14).
    
    """
    df = pd.read_csv(file_csv, index_col="Unnamed: 0")
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

    df.to_csv(file_csv)


def get_support_resistance(file_csv:str, window_size:int=10, threshold:int=0.05):
    """
    Определяет уровни сопротивления и поддержки на основе скользящего среднего
    :param data: pandas.DataFrame с ценами акций
    :param window_size: размер окна скользящего среднего
    :param threshold: пороговое значение отклонения от скользящего среднего для определения уровня
    """
    df = pd.read_csv(file_csv, index_col="Unnamed: 0")
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

    df.to_csv(file_csv)


class Procee:

    def __init__(self, path_launch:str, launch_start:int = 0, launch_end:int = 0, timetravel: str = "5m", func_filter:str = "create_dataset", flag_save: bool = False) -> None:
        self.path_launch = path_launch
        self.launch_start = launch_start
        self.launch_end = launch_end
        self.timetravel = timetravel

        self.name_file_in_launch_csv = "default.csv"

        self.dataset_main = None

        self.time_seconds = int(timetravel.replace(timetravel[-1], "")) * {"m": 60, "H": 60**2, "D":24*60**2}[timetravel[-1]]

        self.timetravel_dict = self.create_timetravel_dict()

        self.procces_dataset(func_filter, flag_save)

        check_df(self.dataset_main)

    @classmethod
    def chit_none_df(cls, df: pd.DataFrame) -> int:
        return(df['open'] == "x").sum()
    

    @classmethod
    def get_none_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df, str):
            df = pd.read_csv(df, index_col="Unnamed: 0")
            df["datetime"] = pd.to_datetime(df['datetime'])

        return df.loc[df['open'] == "x"]


    def create_timetravel_dict(self) -> dict:
        if self.timetravel[-1] == "m":
            time_end = "18:55"

        elif self.timetravel == "1H":
            time_end = "18:00"

        elif self.timetravel == "4H":
            time_end = "15:00"

        elif self.timetravel == "1D":
            time_end, time_start = "00:00", "00:00"
            return {"start": datetime.strptime(time_start, '%H:%M'), "end": datetime.strptime(time_end, '%H:%M')}

        else:
            return {"start": None, "end": None}

        return {"start": datetime.strptime("07:00", '%H:%M'), "end": datetime.strptime(time_end, '%H:%M')}
    
    @dtimer
    def check_datetime(self) -> None:
        print("[INFO] Start check_datetime")
        int_end = self.dataset_main["time_int"].max()
        hour_start, minure_start = [self.timetravel_dict["start"].hour, self.timetravel_dict["start"].minute]
        d = {}
        for i, item in self.dataset_main.iterrows():
            if len(d) != 2:
                d[len(d)] = [i, item]

            elif len(d) == 2:
                d_1, d_2 = list(d.values())
                time_int_1 = d_1[1]["time_int"]
                time_int_2 = d_2[1]["time_int"]

                day_int_1 = d_1[1]["day_int"]
                day_int_2 = d_2[1]["day_int"]

                if abs(time_int_2 - time_int_1) != 1:
                    r = 0
                    if time_int_1 != int_end:
                        d_start = d_1[1]["datetime"]
                    else:
                        if time_int_2 == 1:
                            d_start = d_2[1]["datetime"]
                        else:
                            d_start = d_2[1]["datetime"].replace(hour=hour_start, minute=minure_start)
                            r = -1

                    if day_int_1 != day_int_2:
                        hour_end, minute_end = [self.timetravel_dict["end"].hour, self.timetravel_dict["end"].minute]
                    else:
                        hour_end, minute_end = [d_2[1]["datetime"].hour, d_2[1]["datetime"].minute]

                    while True:
                        d_start_t = d_start + (r + 1) * timedelta(seconds=self.time_seconds)

                        if d_start_t in self.dataset_main["datetime"].values:
                            break

                        time_day_int_t = confert_datetime_to_int_item(d_start_t, self.timetravel_dict, self.timetravel)
              
                        time_t = time_day_int_t["time_int"]
                        day_t = time_day_int_t["day_int"]

                        self.dataset_main.loc[len(self.dataset_main)] = [d_start_t, day_t, time_t] + ["x"] * 5

                        if d_start_t.hour == hour_end and d_start_t.minute == minute_end:
                            if day_int_1 != day_int_2:
                                hour_end, minute_end = [d_2[1]["datetime"].hour, d_2[1]["datetime"].minute]
                                d_start = d_2[1]["datetime"].replace(hour=hour_start, minute=minure_start)
                                r = -1
                                day_int_1 = day_int_2
                                continue
                            break
                        r += 1

                d = {0: d_2, 1: [i, item]}

        self.dataset_main = self.dataset_main.sort_values('datetime', ignore_index=True)
        self.dataset_main.drop(self.dataset_main.loc[self.dataset_main["day_int"] > 5].index, inplace=True)
        count = self.chit_none_df(self.dataset_main)
        print(f"None {count=}")
        print("[INFO] End check_datetime")


    def confert_datetime_to_int(self) -> None:
        self.dataset_main.insert(1, "time_int", 0)
        self.dataset_main.insert(1, "day_int", 0)

        time_start = self.timetravel_dict["start"]
        timetravel_str = self.timetravel

        for i, item in enumerate(self.dataset_main["datetime"]):
            time_s = item.replace(hour=time_start.hour, minute=time_start.minute)
            n = int(timetravel_str.replace(timetravel_str[-1], ""))
            delta = int(abs(item - time_s).total_seconds() // (n*{"m": 60, "H": 60**2, "D":34*60**2}[timetravel_str[-1]]))
            
            self.dataset_main.loc[i, "time_int"] = delta + 1 
            self.dataset_main.loc[i, "day_int"] = item.isoweekday()

    def procces_dataset(self, func_filter, flag_save: bool = False):
        """
        Функця 
        """
        print("[INFO] Start procces_dataset")

        for file in os.listdir():

            if not func_filter in file:
                continue

            if self.launch_start <= int(file.split("_")[-1]) <= self.launch_end:

                file_in_launch_csv = next(filter(lambda x: f".csv" in x, os.listdir(file)))
            
                if file_in_launch_csv.split("-")[-1].replace(".csv", "") == self.timetravel:
                    self.name_file_in_launch_csv = file_in_launch_csv

                    df = pd.read_csv(f"{file}/{file_in_launch_csv}", index_col="Unnamed: 0")
                    df["value"] = df["value"].apply(convert_value)

                    self.dataset_main = df if self.dataset_main is None else pd.concat([self.dataset_main, df], ignore_index=True)

            if int(file.split("_")[-1]) >= self.launch_end:
                break
            
        if not self.dataset_main is None:

            self.dataset_main = procces(self.dataset_main)
            self.confert_datetime_to_int()
            #self.check_datetime()

            if flag_save:
                self.create_launch_dir()

        print("[INFO] End procces_dataset")

    def create_launch_dir(self):
        os.mkdir(self.path_launch)
        os.chdir(self.path_launch)
        self.dataset_main.to_csv(self.name_file_in_launch_csv)

    @classmethod
    def drop_df(cls, file_csv:str) -> None:

        df = pd.read_csv(file_csv, index_col="Unnamed: 0")

        for i, column in enumerate(df.columns):
            if i > 7:
                df = df.drop(column, axis=1)

        df.to_csv(file_csv)


if __name__ == "__main__":
    file = "procces_dataset_launch_1/CNY-5m.csv"
    df = pd.read_csv(file, index_col="Unnamed: 0")
    count = Procee.chit_none_df(df)
    print(f"None {count=}")