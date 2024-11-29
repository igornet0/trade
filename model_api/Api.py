from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as tts
from DProcess import Process
from matplotlib import pyplot as plt
import math
import joblib
import os


def train_test_split(df: pd.DataFrame, n: int = 5) -> tuple[list, list, list, list]:
    X = []
    y = []
    l = []
    for i, item in df.iterrows():
        if len(l) != n:
            if item['open'] != "x":
                l.append([item['day_int'], item['time_int'], float(item['open']), float(item['max']), float(item['min']), float(item['close']), float(item['value'])])
        if len(l) == n:
            if item['open'] == "x":
                continue
            p = []
            for j in l:
                p.extend(j)
            X.append(p)
            y.append([float(item['open']), float(item['max']), float(item['min']), float(item['close']), float(item['value'])])
            l = []
    X = np.array(X)
    y = np.array(y)
    print(Process.chit_none_df(df))
    return tts(X, y, test_size=0.2)


def train_test_split_trade(df: pd.DataFrame) ->tuple[list, list, list, list]:
    pass


def create_dataset_lstm(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


class Api:

    def __init__(self, path_launch, file_path_csv: str, flag_save: bool = True) -> None:

        self.df = pd.read_csv(file_path_csv, index_col="Unnamed: 0")
        self.file_path_csv = file_path_csv

        self.path_launch = path_launch
        self.flag_save = flag_save

        self.model = None


    def train_model_pred_dataset(self):
        """
        Функция на основе датасета обучает модель
        """

        X_train, X_test, y_train, y_test = train_test_split(self.df)

        self.model = Sequential([
                            Dense(35, activation='relu', input_shape=(35,)),
                            Dense(5)
                                ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.model.fit(X_train, y_train, epochs=1000)

        self.error(X_test, y_test)

        if self.flag_save:
            self.name_model = "model_forest"
            self.create_launch_dir()

    
    def lstm_train(self, look_back: int = 15):
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = min_max_scaler.fit_transform(self.df['close'].values.reshape(-1, 1))  
        self.show(self.df['close'])

        train_size = int(len(dataset) * 0.9)

        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        print(len(train), len(test))

        x_train, y_train = create_dataset_lstm(train, look_back)
        x_test, y_test = create_dataset_lstm(test, look_back)

        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        self.model = Sequential()
        self.model.add(LSTM(20, input_shape=(1, look_back)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)

        trainPredict = self.pred_model(x_train)
        testPredict = self.pred_model(x_test)

        trainPredict = min_max_scaler.inverse_transform(trainPredict)
        trainY = min_max_scaler.inverse_transform([y_train])
        testPredict = min_max_scaler.inverse_transform(testPredict)
        testY = min_max_scaler.inverse_transform([y_test])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

        self.show([min_max_scaler.inverse_transform(dataset), trainPredictPlot, testPredictPlot])

        if self.flag_save:
            self.name_model = "model_lstm"
            self.create_launch_dir()

    
    def train_model_trade(self):

        X_train, X_test, y_train, y_test = train_test_split_trade(self.df)

        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(6,)),
            Dense(64, activation='relu'),
            Dense(6)
        ])

        # Компиляция модели
        self.model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

        # Обучение нейронной сети
        history = self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        print(history)

        if self.flag_save:
            self.name_model = "model_trade"
            self.create_launch_dir()

    
    def test_lstm(self, look_back: int = 15):
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = min_max_scaler.fit_transform(self.df['close'].values.reshape(-1, 1))  
        self.show(self.df['close'])

        train_size = int(len(dataset) * 0.9)

        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        print(len(train), len(test))

        x_test, y_test = create_dataset_lstm(test, look_back)
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
        testPredict = self.pred_model(x_test)

        testPredict = min_max_scaler.inverse_transform(testPredict)
        testY = min_max_scaler.inverse_transform([y_test])

        # calculate root mean squared error
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))

        if self.flag_save:
            self.name_model = "model_lstm"
            self.create_launch_dir()


    def error(self, X_test, y_test) -> float:
        y_pred = self.pred_model(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print('Mean Squared Error:', mse)
        return mse
    
    def model_save(self):
        joblib.dump(self.model, f"{self.name_model}.joblib")


    def model_load(self, file_path_model):
        self.model = joblib.load(file_path_model)


    def pred_model(self, X: np.array):
        return self.model.predict(X)


    def create_launch_dir(self):
        os.mkdir(self.path_launch)
        os.chdir(self.path_launch)
        if not self.model is None:
            self.model_save()


    def show(self, object: object):
        if isinstance(object, list):
            plt.plot(*object)
        else:
            plt.plot(object)

        plt.show()


def proces(api: Api):
    df = api.df
    for i, item in df.iterrows():
        if item['open'] == "x":
            if i < 7:
                continue
            l_df = df.iloc[i-6:i-1]
            f = False
            for j, item_l in l_df.iterrows():
                if item_l["value"] == "x":
                    f = True
                    break
            if f:
                continue
                
            l = list(map(lambda x: [x[1]['day_int'], x[1]['time_int'], float(x[1]['open']), float(x[1]['max']), float(x[1]['min']), float(x[1]['close']), float(x[1]['value'])], l_df.iterrows()))
            p = []
            for j in l:
                p.extend(j)
            y = api.pred_model(np.array([p]))
            for j, it in enumerate(l_df.columns[3:]):
                result = round(y[0][j], 2)
                df.at[i, it] = result
    
    print(Process.chit_none_df(df))
    df.to_csv(api.file_path_csv)