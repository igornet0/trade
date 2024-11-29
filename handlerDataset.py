import os 
from os import mkdir, chdir
from DProcess import *
from indicaters import *
import pandas as pd
import pathlib

class Handler:

    def __init__(self, path_launch:str, path_data: str, DEBUG=False) -> None:
        self.main_dir = os.getcwd()
        self.path_launch = path_launch
        self.path_data = path_data
        self.flag = False
        
        self.DEBUG = DEBUG

        self.data_tree = dict()

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

    def create_launch_dir(self):
        if self.flag:
            return
        mkdir(f"{self.main_dir}/{self.path_launch}")
        self.flag = True
        #chdir(f"{self.main_dir}/{self.path_data}/{self.path_launch}")

    def remove_launch_dir(self):
        #chdir(self.main_dir)
        if not self.flag:
            return
        rmtree(f"{self.main_dir}/{self.path_launch}")
        self.flag = False

    def set_default_column(self, dataset):
        columns = ["date", "Open", "high", "low", "close", "Volume USDT"]
        try:
            new_df = dataset[columns].rename(columns={'Volume USDT': 'volume'}).rename(columns={'Volume': 'volume'})
        except:
            new_df = dataset.rename(columns={'Volume USDT': 'volume'}).rename(columns={'Volume': 'volume'})

        return new_df

    def get_data_from_csv(self, file: str, timetravel:str = None) -> dict[str,pd.DataFrame]:
        print(f"[INFO] Start get_data_from_csv {file=}") if self.DEBUG else None

        if not ".csv" in file:
            raise ValueError("File must be a valid CSV file")

        if timetravel is None or file.split("-")[-1].replace(".csv", "") == timetravel:
            self.name_file_in_launch_csv = file
            df = pd.read_csv(f"{self.main_dir}/{self.path_data}/{file}")
            df = self.set_default_column(df)
            df.name = file.replace(".csv", "")
            #df["Volume USDT"] = df["Volume USDT"].apply(convert_value)

            if self.data_tree.setdefault(df.name, None) is None:
                self.data_tree[df.name] = df
            else:
                self.data_tree[df.name] = pd.concat([self.data_tree[df.name], df], ignore_index=True)

        print("[INFO] End get_data_from_csv") if self.DEBUG else None
        return self.data_tree

    def get_data_tree_path(self):
        for file in os.listdir(f"{self.main_dir}/{self.path_data}"):
            self.get_data_from_csv(file)

    def get_data_traa(self) -> dict[str, pd.DataFrame]:
        return self.data_tree

    def process_dataset(self, timetravel:str = "5m", get_data: bool = False, flag_save: bool = False) -> dict[str,pd.DataFrame]:
        
        print("[INFO] Start process_dataset") if self.DEBUG else None

        if get_data:
            self.get_data_tree_path()

        if len(self.data_tree.keys()) > 0:
            for active, dataset in self.data_tree.items():
                processD = Process(dataset, timetravel)
                processD.process()
                processD.confert_datetime_to_int()
                processD.check_datetime()
                
                dataset = processD.get_dataset()
                if flag_save:
                   self.safe(dataset, active)
                
                self.data_tree[active] = dataset

        chdir(self.main_dir)
        print("[INFO] End process_dataset") if self.DEBUG else None
        return self.data_tree

    def indecaters_add(self, get_data: bool = False, flag_save: bool = False) -> dict[str,pd.DataFrame]:
        print("[INFO] Start indecaters_add") if self.DEBUG else None

        if get_data:
            self.get_data_tree_path()

        if len(self.data_tree.keys()) > 0:
            for active, dataset in self.data_tree.items():
                dataset = Indicater.calculate_technical_indicators(dataset.copy(), SMAtimeperiod=10, RCItimeperiod=14)
                if flag_save:
                    self.safe(dataset, active)
                
                self.data_tree[active] = dataset

        print("[INFO] End indecaters_add") if self.DEBUG else None
        return self.data_tree

    def split_timetravel(self, timetravel: str, get_data: bool = False, flag_save: bool = False) -> pd.DataFrame:
        print("[INFO] Start split_timetravel") if self.DEBUG else None

        if get_data:
            self.get_data_tree_path()

        if len(self.data_tree.keys()) > 0:
            n = 0
            for active, dataset in self.data_tree.items():
                process = Process(dataset, timetravel)
                dataset = process.split_df()
                n += 1
                print(f"[INFO] Processing dataset ready {n}/{len(self.data_tree.values())}")
                if flag_save:
                    self.safe(dataset, f"{active}-{timetravel}")

                self.data_tree[active] = dataset

        print("[INFO] End split_timetravel") if self.DEBUG else None
        return self.data_tree

    def safe(self, dataset, name="data"):
        self.create_launch_dir()
        dataset.to_csv(f"{self.main_dir}/{self.path_launch}/{name}.csv")
        print(f"[INFO] Save {name}") if self.DEBUG else None


