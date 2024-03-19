from web_api.Api import Web_api
from model_api.Api import Api, proces
from sys import argv
import os
from datetime import datetime
from dataset_procces import *

def create_dataset(args: list):
    #python main.py create_dataset 1000 
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
            api.path_launch =  f"data/{l[0]}_launch_{n+1}"

        if len(data) >= counter:
            break

        api.restart()
        datetime_last, data = api.generate(datetime_first=datetime_last)
        
    # elif len(args) == 3:
    #     api.generate_for_df(Procee.get_none_df(args[2]))


def create_model_dataset(args: list):
    #python main.py create_model_dataset procces_dataset_launch_3/CNY-5m.csv
    api = Api(*args)
    api.train_model_pred_dataset()


def lstm_train(args: list):
    #python main.py lstm_train procces_dataset_launch_2/CNY-5m.csv
    api = Api(*args)
    api.lstm_train()


def load_model_dataset(args: list):
    #python main.py load_model_dataset procces_dataset_launch_2/CNY-5m.csv create_model_dataset_launch_2/model_forest.joblib
    #python main.py load_model_dataset procces_dataset_launch_2/CNY-5m.csv lstm_train_launch_2/model_lstm.joblib
    model_path = args.pop(2)
    api = Api(*args, flag_save=False)
    api.model_load(model_path)

    if "forest" in model_path:
        proces(api)
    elif "lstm" in model_path:
        api.test_lstm()


def procces_dataset(args: list):
    #python main.py procces_dataset 1 10 5m
    p = Procee(*args)
    p.create_launch_dir()


def drop_df(args: list):
    #python main.py drop_df procces_dataset_launch_2/CNY-5m.csv
    args.pop(0)
    Procee.drop_df(*args)


def test_atr(args: list):
    #python main.py test_atr procces_dataset_launch_2/CNY-5m.csv
    args.pop(0)
    get_atr(*args)


def test_support_resistance(args: list):
    #python main.py test_support_resistance procces_dataset_launch_2/CNY-5m.csv
    args.pop(0)
    get_support_resistance(*args)


def test_stochastic_oscillator(args: list):
    #python main.py test_stochastic_oscillator procces_dataset_launch_2/CNY-5m.csv
    args.pop(0)
    get_stochastic_oscillator(*args)


def test_trend(args: list):
    #python main.py test_trend procces_dataset_launch_2/CNY-5m.csv
    args.pop(0)
    get_trend(*args)


def main(data:list) -> None:
    func_a = globals()[data[1]]
    args = [data[0]] + list(map(lambda x: int(x) if x.isdigit() else x, data[2:]))
    func_a(args)

    
if __name__ == "__main__":
    skript, *l = argv

    n = 0
    for file in os.listdir("data"):
        if l[0] in file:
            if int(file.split("_")[-1]) >= n:
                n = int(file.split("_")[-1])

    path_launch = f"data/{l[0]}_launch_{n+1}"
    l.insert(0, (path_launch, n+1))
    print("[INFO] START main")
    main(l)
    print("[INFO] END main")
