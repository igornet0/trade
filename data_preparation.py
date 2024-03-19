import numpy as np
import pandas as pd

from avg import moving_avg

def data() -> list:
    lst = []
    with open(f"data/{input('file name: ')}.txt", "r") as file:
        for item in file.readlines():
            item = item.split()
            value = item[-1]
            if "K" in value:
                value = value.replace("K", "")
                value = float(value) * 1000
            elif "M" in value:
                value = value.replace("M", "")
                value = float(value) * (10**6)
            elif "B" in value:
                value = value.replace("B", "")
                value = float(value) * (10**9)
            item[-1] = value
            lst.append(list(map(lambda x: float(x), item)))
    return lst

def AVG(lst: list, avg_n: int) -> list:
    avd_lst = []
    for i in lst:
        avd_lst.append(i[3])
    avd_lst = moving_avg(avd_lst, avg_n)

    avd_lst = np.concatenate([np.array([0] * (len(lst) - len(avd_lst))), avd_lst])
    #array = array + avd_lst5 + avd_lst10 + avd_lst15 + avd_lst20
    for i in range(len(lst)):
        lst[i].append(avd_lst[i])

    return lst

def main():
    lst = data()
    lst = AVG(lst, 5)
    lst = AVG(lst, 20)
    lst = AVG(lst, 50)
    lst = AVG(lst, 100)
    df = pd.DataFrame(np.array(lst), columns = ['open','maxm','minm','close','value', 'avg5','avg20','avg50','avg100'])
    df.to_csv("data/full_GDM3.csv")

if __name__ == "__main__":
    main()