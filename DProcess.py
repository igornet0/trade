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
   
class Process:

    def __init__(self, dataset: pd.DataFrame = None, timetravel: str = "5m") -> None:

        self.timetravel = timetravel
        self.time_seconds = int(timetravel.replace(timetravel[-1], "")) * {"m": 60, "H": 60**2, "D":24*60**2}[timetravel[-1]]

        self.timetravel_dict = self.create_timetravel_dict()

        self.dataset = dataset

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
    
    def get_dataset(self):
        return self.dataset.copy()

    def process(self, dataset:pd.DataFrame = None) -> pd.DataFrame:
        if dataset is not None:
            self.dataset = dataset.copy()

        self.dataset["datetime"] = pd.to_datetime(self.dataset['datetime'])
        self.dataset["datetime"] = self.dataset["datetime"].apply(self.check_time_minute)

        self.dataset.drop(self.dataset.loc[self.dataset["datetime"].dt.hour > 18].index, inplace=True)

        self.dataset = self.dataset.sort_values('datetime', ignore_index=True)
        self.dataset = self.dataset.drop_duplicates(subset='datetime', ignore_index=True)

        return self.dataset

    @dtimer
    def check_datetime(self, dataset: pd.DataFrame = None) -> pd.DataFrame:
        if dataset is not None:
            self.dataset = dataset.copy()

        print("[INFO] Start check_datetime")

        int_end = self.dataset["time_int"].max()
        hour_start, minure_start = [self.timetravel_dict["start"].hour, self.timetravel_dict["start"].minute]

        d = {}
        for i, item in self.dataset.iterrows():
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

                        if d_start_t in self.dataset["datetime"].values:
                            break

                        time_day_int_t = self.confert_datetime_to_int_item(d_start_t)
              
                        time_t = time_day_int_t["time_int"]
                        day_t = time_day_int_t["day_int"]

                        self.dataset.loc[len(self.dataset)] = [d_start_t, day_t, time_t] + ["x"] * 5

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

        self.dataset = self.dataset.sort_values('datetime', ignore_index=True)
        self.dataset.drop(self.dataset.loc[self.dataset["day_int"] > 5].index, inplace=True)
        count = self.chit_none_df(self.dataset)
        print(f"None {count=}")
        print("[INFO] End check_datetime")

        return self.dataset

    def confert_datetime_to_int(self, dataset: pd.DataFrame = None) -> pd.DataFrame:
        if not dataset is None:
            self.dataset = dataset.copy()

        self.dataset.insert(1, "time_int", 0)
        self.dataset.insert(1, "day_int", 0)

        time_start = self.timetravel_dict["start"]
        timetravel_str = self.timetravel

        for i, item in enumerate(self.dataset["datetime"]):
            time_s = item.replace(hour=time_start.hour, minute=time_start.minute)
            n = int(timetravel_str.replace(timetravel_str[-1], ""))
            delta = int(abs(item - time_s).total_seconds() // (n*{"m": 60, "H": 60**2, "D":34*60**2}[timetravel_str[-1]]))
            
            self.dataset.loc[i, "time_int"] = delta + 1 
            self.dataset.loc[i, "day_int"] = item.isoweekday()

        return self.dataset

    def confert_datetime_to_int_item(self, item:datetime) -> dict:
        try:
            time_start = self.timetravel_dict["start"]
            timetravel_str = self.timetravel

            time_s = item.replace(hour=time_start.hour, minute=time_start.minute)
            n = int(timetravel_str.replace(timetravel_str[-1], ""))
            delta = int(abs(item - time_s).total_seconds() // (n*{"m": 60, "H": 60**2, "D":34*60**2}[timetravel_str[-1]]))

            return {"time_int": delta + 1, "day_int": item.isoweekday()}
        except TypeError:
            raise ValueError("Неверный формат даты")

    def check_time_minute(self, d: datetime) -> datetime:
        func_temp = lambda x: x.time().minute % int(self.timetravel.replace(self.timetravel[-1], ""))
        return d.replace(minute=d.time().minute - (func_temp(d))) if func_temp(d) != 0 else d

    @classmethod
    def drop_df(cls, file_csv:str) -> None:
        """
        Удаляем лишние столбцы, файл csv становится стандартным
        :param file_csv: путь к файлу csv
        """
        df = pd.read_csv(file_csv, index_col="Unnamed: 0")

        for i, column in enumerate(df.columns):
            if i > 7:
                df = df.drop(column, axis=1)

        df.to_csv(file_csv)


    def split_df(self, dataset: pd.DataFrame = None, timetravel: str = None) -> pd.DataFrame:
        if not timetravel is None:
            self.timetravel = timetravel
        if not dataset is None:
            self.dataset = dataset.copy()

        dataset = self.dataset.copy()
        print(f"[INFO] Start split_df {self.dataset.name}")
        dataset["date"] = pd.to_datetime(dataset['date'])
        dataset = dataset.sort_values('date', ignore_index=True)

        if self.timetravel[-1] == "D":
            result = dataset.groupby(dataset['date'].dt.date)['date'].agg(['first', 'last']).reset_index()

        buffer_data = {"date": None, "open":None,"high":0,"low":100000,"close":None,"volume":0}

        new_dataset = pd.DataFrame(columns=buffer_data.keys())

        for i, item in enumerate(dataset["date"]):
            buffer_data["volume"] += dataset["volume"].iloc[i]

            buffer_data["high"] = max(dataset["high"].iloc[i], buffer_data["high"])
            buffer_data["low"] = min(dataset["low"].iloc[i], buffer_data["low"])  

            param = 0
            if self.timetravel[-1] == "m":
                if item.minute % int(self.timetravel.replace("m", "")) == 0:
                    param = 1
                    buffer_data["date"] = item.strftime('%Y-%m-%d %H:%M')

            elif self.timetravel[-1] == "H":
                if i < len(dataset) - 1 and dataset["date"].iloc[i + 1].hour != item.hour:
                    param = 1
                    buffer_data["date"] = item.strftime('%Y-%m-%d %H:00')
                
            elif self.timetravel[-1] == "D":
                if buffer_data["open"] is None or buffer_data["close"] is None:
                    for index, row in result.iterrows():
                        if row["first"] == item:
                            buffer_data["date"] = item.strftime('%Y-%m-%d')
                            buffer_data["open"] = None
                            break
                        if row["last"] == item:
                            param = 1
                            break
            if param == 0:
                if buffer_data["open"] is None:
                    buffer_data["open"] = dataset["open"].iloc[i]

            else:
                if buffer_data["open"] is None:
                    buffer_data["open"] = dataset["open"].iloc[i]

                buffer_data["close"] = dataset["close"].iloc[i]

                if None in buffer_data.values():
                    raise ValueError(f"None value {buffer_data}")

                dt = pd.DataFrame([buffer_data.values()], columns=buffer_data.keys())
                new_dataset = pd.concat([dt, new_dataset if not new_dataset.empty else None], ignore_index=True)
                buffer_data = {"date": None,"open":None,"high":0,"low":100000,"close":None,"volume":0}


        print("[INFO] End split_df")

        return new_dataset