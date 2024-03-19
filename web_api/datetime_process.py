from datetime import datetime
import time

import pytesseract
from PIL import Image

mounth = {"Дек": 12, "Ноя": 11, "Окт": 10, "Сен": 9, "Авг": 8, "Июл": 7, 
                "Июн": 6, "Май": 5, "Апр": 4, "Мар": 3, "Фев": 2, "Янв": 1}

def image_to_text(filename:str) -> str:
        img = Image.open(filename)
        bbox = img.getbbox()
        cropped_img = img.crop(bbox)
        cropped_img = cropped_img.convert('RGB')

        text = pytesseract.image_to_string(cropped_img, lang='rus')

        return text

def str_to_datatime(string:str) -> datetime:
    try:
        result = {}
        string_l = string.split()
        string_l[2] = string_l[2].replace("'", '')
        string_l[2] = string_l[2].replace('"', '')
        string_l[1] = mounth[string_l[1]]
        string_l[2] = "20" + string_l[2]
        result["year"] = int(string_l[2])
        result["month"] = string_l[1]
        result["day"] = int(string_l[0])

        if len(string_l) == 4:
            time_l = list(map(int, string_l[-1].split(":")))
            result["hour"] = time_l[0]
            result["second"] = time_l[1]
        else:
            raise TypeError(f"format was expected '%D %M '%Y %H:%m', received {string}")

        return datetime(**result)

    except Exception as e:
        print("ERROR datetime")
        print(f"Error: {e}")
        print("Datetime: ", string)
        string = input("New Datetime: ")
        time.sleep(5)
        if string == "exit":
            return False
        return str_to_datatime(string)
    

if __name__ == "__main__":
    print(str_to_datatime("26 Дек '23 20:19"))
    print(str_to_datatime("26 Дек '23"))
