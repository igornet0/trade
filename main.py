from web_api.Api import Web_api
from model_api.Api import Api, proces
from sys import argv
import os
from datetime import datetime
from DProcess import *
from indicaters import *
from handlerDataset import Handler

def parser_create_dataset(args: list):
    #python main.py create_dataset 1000 ?True data  
    #python main.py create_dataset 1 2024-03-13 14:15:00
    URL = "https://lk.bcs.ru/terminal"
    login = "89833087191"
    password = "1607Igor"
    path_launch, n = args[0]
    counter = args[1]

    save = True
    tick = 0

    main_dir = os.path.abspath(__file__).replace(skript, "")

    api = Web_api(main_dir=main_dir, path_launch=path_launch, save=save, counter=counter, tick=tick)
    
    api.start_web(URL, login, password)

    datetime_first = None if len(args) < 3 else datetime.strptime(args[2], "%Y-%m-%d %H:%M:%S")
    datetime_last, data = api.generate(datetime_first=datetime_first)

    while True:
        counter -= len(data)
        if len(data) < 10 and save:
            datetime_last = None
            api.remove_launch_dir()
        else:
            n += 1
            api.path_launch =  f"data/{l[0]}_launch_{n+1}" #data deleted

        if len(data) >= counter:
            break

        api.restart()
        datetime_last, data = api.generate(datetime_first=datetime_last)
        
    # elif len(args) == 3:
    #     api.generate_for_df(Procee.get_none_df(args[2]))

def create_dataset(args: list):
    #python main.py data create_dataset ?1000 ?True ?data
    pass



def create_model_dataset(args: list):
    #python main.py create_model_dataset process_dataset_launch_3/CNY-5m.csv
    api = Api(*args)
    api.train_model_pred_dataset()


def lstm_train(args: list):
    #python main.py lstm_train process_dataset_launch_2/CNY-5m.csv
    api = Api(*args)
    api.lstm_train()


def load_model_dataset(args: list):
    #python main.py load_model_dataset process_dataset_launch_2/CNY-5m.csv create_model_dataset_launch_2/model_forest.joblib
    #python main.py load_model_dataset process_dataset_launch_2/CNY-5m.csv lstm_train_launch_2/model_lstm.joblib
    model_path = args.pop(2)
    api = Api(*args, flag_save=False)
    api.model_load(model_path)

    if "forest" in model_path:
        proces(api)
    elif "lstm" in model_path:
        api.test_lstm()


def process_dataset(args: list):
    #python main.py process_dataset data ?5m ?True
    #Process.create_launch_dir(*args)
    path_launch, _ = args.pop(0)
    path_data = args.pop(0)
    timetravel = None
    flag_save = False
    path_filter = None
    if len(args) > 0:
        timetravel = args.pop(0)
    if len(args) > 0:
        flag_save = args.pop(0)
    if len(args) > 0:
        path_filter = args.pop(0)

    hadler_d = Handler(path_launch, path_data)
    hadler_d.process_dataset(timetravel, flag_save, path_filter)


def indecaters_add(args: list):
    #python main.py data indecaters_add process_dataset_launch_2/CNY-5m.csv ?True
    #python main.py data indecaters_add data/process_dataset_launch_1/
    path_launch, _ = args.pop(0)
    path_data = args.pop(0)
    flag_save = False
    DEBUG = False

    if len(args) > 0:
        flag_save = args.pop(0)

    if len(args) > 0:
        DEBUG = args.pop(0)

    hadler_d = Handler(path_launch, path_data, DEBUG)
    hadler_d.indecaters_add(True, flag_save=flag_save)


def split_df(args: list):
    #python main.py split_df data indecaters_add_1/CNY-5m.csv ?5m ?True
    path_launch, _ = args.pop(0)
    path_data = args.pop(0)
    timetravel = None
    flag_save = False
    path_filter = "launch"
    if len(args) > 0:
        timetravel = args.pop(0)
    if len(args) > 0:
        flag_save = args.pop(0)
    if len(args) > 0:
        path_filter = args.pop(0)

    hadler_d = Handler(path_launch, path_data)
    hadler_d.split_timetravel(timetravel, flag_save, path_filter)


def drop_df(args: list):
    #python main.py drop_df process_dataset_launch_2/CNY-5m.csv
    args.pop(0)
    Process.drop_df(*args)

def allprocess_dataset(args: list):
    #python main.py allprocess_dataset data/process_dataset_launch_1/ 5m ?True
    #Process.create_launch_dir(*args)
    path_launch, _ = args.pop(0)
    path_data = args.pop(0)
    flag_save = False
    timetravel = args.pop(0)
    DEBUG = False

    if len(args) > 0:
        flag_save = args.pop(0)

    if len(args) > 0:
        DEBUG = args.pop(0)

    hadler_d = Handler(path_launch, path_data, DEBUG)

    hadler_d.split_timetravel(timetravel, True, False)

    hadler_d.process_dataset(timetravel=timetravel, flag_save=False)

    hadler_d.indecaters_add(flag_save=flag_save)


def main(data:list) -> None:
    func_a = globals()[data[1]]
    args = [data[0]] + list(map(lambda x: int(x) if x.isdigit() else x, data[2:]))
    func_a(args)

    
if __name__ == "__main__":
    skript, *l = argv
    n = 0
    data_pah = l.pop(0)
    for file in os.listdir(f"{data_pah}"):
        if l[0] in file:
            if int(file.split("_")[-1]) >= n:
                n = int(file.split("_")[-1])

    path_launch = f"{data_pah}/{l[0]}_launch_{n+1}"
    l.insert(0, (path_launch, n+1))
    print("[INFO] START main")
    main(l)
    print("[INFO] END main")
