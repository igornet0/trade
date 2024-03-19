from os import listdir
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time

def dtimer(func):

    def start(*args, **kwargs):
        start = time.time()
        data = func(*args, **kwargs)
        end = time.time()
        timer = round(end - start, 3)
        print(f"{timer=}")
        return data

    return start


def convert_value(value: str) -> float:
    if value == "x":
        return value
    
    value_int = {"K": 10**3, "M": 1, "B": 10**(-3)}
    
    return round(float(value.replace(value[-1], "")) / value_int[value[-1]], 2)


def check_start_end_time(l_se:list, d:datetime) -> bool:
    for datetime in l_se:
        if datetime.time() == d.time():
            return True
    return False


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
        
def check_time_minute(d: datetime, timetravel:str = "5m") -> datetime:
    func_temp = lambda x: x.time().minute % int(timetravel.replace(timetravel[-1], ""))

    return d.replace(minute=int(input(f"{d=}\nminute = "))) if func_temp(d) != 0 else d


def procces(df:pd.DataFrame, timetravel:str = "5m") -> pd.DataFrame:
    df["datetime"] = pd.to_datetime(df['datetime'])
    df["datetime"] = df["datetime"].apply(check_time_minute, args=(timetravel, ))

    df.drop(df.loc[df["datetime"].dt.hour > 18].index, inplace=True)

    df = df.sort_values('datetime', ignore_index=True)
    df = df.drop_duplicates(subset='datetime', ignore_index=True)
    return df



def data_by_day_generator(df):
    dates = df['datetime'].dt.date.unique()

    for date in dates:
        daily_data = df[df['datetime'].dt.date == date]
        yield daily_data


class Prepare:

    def __init__(self, path_df: str) -> None:
        self.path_df = path_df

    def get_files_df(self, filter_path: str = "create_dataset"):
        files_df = {}
        for path in listdir(self.path_df):
            if filter_path in path:
                for file in listdir(f"{self.path_df}/{path}"):
                    if ".csv" in file:
                        files_df.setdefault(file.replace(".csv", ""), []).append(f"{path}/{file}")

        return files_df
    

    def get_time_range(self) -> dict:
        time_start = "07:00"

        if self.timetravel[-1] == "m":
            time_end = "18:55"

        elif self.timetravel == "1H":
            time_end = "18:00"

        elif self.timetravel == "4H":
            time_end = "15:00"

        elif self.timetravel == "1D":
            time_end, time_start = "00:00", "00:00"
    
        time_range = pd.date_range(start=time_start, end=time_end, freq=self.timetravel.replace("m", "T"))

        return time_range
    
    @dtimer
    def conncat_missing_rows(self, df):
        
        time_range = self.get_time_range()

        df = df[(df['datetime'].dt.time >= time_range[0].time()) & (df['datetime'].dt.time <= time_range[-1].time())]

        missing_rows = []
    
        for day_df in data_by_day_generator(df):
            date = day_df['datetime'].dt.date.unique()[0]
            for time in time_range:
                if not day_df["datetime"].astype(str).str.contains(time.strftime('%H:%M:%S')).any():
                    time = time.time()
                    new_dt = datetime(date.year, date.month, date.day, time.hour, time.minute, time.second)

                    new_row = {'datetime': new_dt}
                    for col in day_df.columns[1:]:
                        new_row[col] = 'x'

                    missing_rows.append(new_row)

        return pd.concat([df, pd.DataFrame(missing_rows)]).sort_values('datetime', ignore_index=True)


    def concat_df_process(self):
        """
        Merge all the CSV files from the data folder to a single one,
        and remove duplicate rows.

        Then, add missing rows based on the time travel interval
        of the current file.

        Finally, save the resulting DataFrame to a new CSV file with the
        same name as the original one.
        """
        files_df = self.get_files_df()

        for name, file_list in files_df.items():
            # Read all the CSV files from the data folder
            df = pd.concat(
                [pd.read_csv(f"data/{file}", index_col="Unnamed: 0") for file in file_list]
            )
            # Remove duplicate rows
            df = df.drop_duplicates()

            # Convert the 'datetime' column to datetime format
            df['datetime'] = pd.to_datetime(df['datetime'])

            # Get the time travel interval of the current file
            self.timetravel = name.split("-")[-1]

            # Add missing rows based on the time travel interval
            df = self.conncat_missing_rows(df)

            # Save the resulting DataFrame to a new CSV file
            df.to_csv(f'{name}.csv', index=False)


if __name__ == "__main__":
    Prepare("data").concat_df_process()
