import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    file = "procces_dataset_launch_2/CNY-5m.csv"
    df = pd.read_csv(file, index_col="Unnamed: 0")

    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    object = min_max_scaler.fit_transform(df['close'].values.reshape(-1, 1))[:100]


    if isinstance(object, list):
        plt.plot(*object)
    else:
        plt.plot(object)

    plt.show()