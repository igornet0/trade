from selenium.webdriver.common.by import By
import undetected_chromedriver as uc

import pyautogui

import pandas as pd
import time, base64
from datetime import datetime
from shutil import rmtree
from os import remove, mkdir, chdir

from web_api.datetime_process import *

dict_timetravel = {"1Д":"1D", "4Ч":"4H", "1Ч":"1H"}


def save_data(data: pd.DataFrame, file:str) -> None:
    data.to_csv(file)
    return data

class Web_api:

    def __init__(self, main_dir:str, path_launch:str, tick:int = 1, save:bool = False, counter:int = 0, DEBUG=False) -> None:
        self.main_dir = main_dir
        self.path_launch = path_launch

        self.save = save
        
        self.counter = counter
        self.tick = tick

        self.xpath = {}
        self.xpath_vxod = {}

        self.x_start  = None
        self.log_direction = []
        self.log_del = []

        self.DEBUG = DEBUG

        self.default_xpath()

    
    def default_xpath(self):
        self.add_xpath("login", "/html/body/div[1]/div[3]/div/div[2]/div/div[2]/div/form/div[1]/div/div/input")
        self.add_xpath("password", "/html/body/div[1]/div[3]/div/div[2]/div/div[2]/div/form/div[2]/div/div/input")
        self.add_xpath("click_login", "/html/body/div[1]/div[3]/div/div[2]/div/div[2]/div/form/div[3]/button[1]")
        self.add_xpath("frame", "/html/body/div[1]/div[3]/div[3]/div[2]/div/div/div[3]/div/div[1]/div[2]/div/div/div[2]/div/div/div[2]/iframe")
        self.add_xpath("filename", "/html/body/div[1]/div[3]/div[3]/div[2]/div/div/div[3]/div/div/div[2]/div/div/div[1]/div[1]/div/div/div/div/div[2]/label[1]/span/div/span")
        self.add_xpath("timetravel", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[1]/div[1]/div[3]")
        self.add_xpath("datetime", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[4]/td[2]/div/canvas[2]")
        self.add_xpath("open", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[2]/div/div[2]/div[2]")
        self.add_xpath("max",  "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[2]/div/div[3]/div[2]")
        self.add_xpath("min", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[2]/div/div[4]/div[2]")
        self.add_xpath("close", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[1]/div/div[2]/div/div[5]/div[2]")
        self.add_xpath("value", "/html/body/div[2]/div[3]/div/div[1]/div[2]/table/tr[1]/td[2]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]/div")
            

    def start_web(self, URL:str = None, login: str = None, password: str = None):
        options = uc.ChromeOptions() 
        self.driver = uc.Chrome(use_subprocess=True, options=options) 
        self.driver.get(URL) 
        self.URL = URL
        self.login = login
        self.password = password
        self.entry()

    def end_web(self):
        self.driver.close()

    def restart(self):
        self.start_web(self.URL, self.login, self.login)


    def entry(self):
        n = 5
        while True:
            try:
                if self.DEBUG:
                    input("Chekpoint #1")
                self.driver.find_elements(By.XPATH, self.xpath_vxod["login"])[0].send_keys(self.login)
                self.driver.find_elements(By.XPATH, self.xpath_vxod["password"])[0].send_keys(self.password)
                self.driver.find_elements(By.XPATH, self.xpath_vxod["click_login"])[0].click() 
                return True
            except:
                time.sleep(3)
                n -= 1
                if n == 0:
                    return False


    def switch_frame(self):
        if self.DEBUG:
            input("Chekpoint #2")
        time.sleep(3)

        frame = self.driver.find_element(By.XPATH, self.xpath_vxod["frame"])
        self.driver.switch_to.frame(frame)  

        self.file = f"{self.get_filename()}-{self.get_timetravel()}.csv"
  
    def add_xpath(self, key:str, xpath:str):
        if key in ["login", "password", "click_login", "frame", "del_datetime", "filename", "timetravel"]:
            self.xpath_vxod[key] = xpath
        else:
            self.xpath[key] = xpath

    def get_element(self) -> dict:
        data_d = {}
        for key, xpath in self.xpath.items():
            try:
                element = self.driver.find_element(By.XPATH, xpath)
                if key == "datetime":
                    result = self.get_element_datetime()
                    if not result:
                        return data_d
                else:
                    while not element.text:
                        element = self.driver.find_element(By.XPATH, xpath)

                    result = element.text
                data_d[key] = result

            except Exception as e:
                print(e)
                break

        return data_d
    
    def get_timetravel(self):
        element = self.driver.find_elements(By.XPATH, self.xpath_vxod["timetravel"])[0]
        return dict_timetravel[element.text] if not element.text.isdigit() else element.text + "m"

    def get_filename(self):
        element = self.driver.find_elements(By.XPATH, self.xpath_vxod["filename"])[0]
        return element.text



    def generate(self, direction:str = "left", datetime_first: datetime = None) -> pd.DataFrame:
        try:
            data = pd.DataFrame(columns=self.xpath.keys())

            datetime_first = datetime.strptime(input("datetime_first: "), "%Y-%m-%d %H:%M:%S") if datetime_first else datetime_first

            self.switch_frame()

            if self.save:
                self.create_launch_dir()

            if not datetime_first is None:
                self.searh_datetime(datetime_first)
            
            for _ in range(self.counter):
                data_d = self.get_element()
                if len(data_d) <= 5:
                    break
                print(len(data)+1, data_d)
                data.loc[len(data)] = data_d

                pyautogui.press(direction)
                time.sleep(self.tick)
        except Exception as e:
            print(e)

        finally:
            self.end_web()
            last_datetime = data['datetime'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
            print(f"[INFO genetate] Last datetime = {last_datetime}")

            return (last_datetime, save_data(data)) if self.save else (last_datetime, data)
        
    def move_cursor(self, direction):
        pyautogui.moveTo(self.x_start, self.y_start)
        move = None
        if len(self.log_direction) > 4:
            if sum(self.log_direction) == 23:
                self.log_del.append(23)
                if len(self.log_del) > 5:
                    return False
            elif sum(self.log_direction) == 20:
                self.log_del = []
            self.log_direction = []
        self.log_direction.append(len(direction))

        if direction == "left_fast":
            move = self.x_end - self.x_start

        elif direction == "left_middle":
            move = (self.x_end - self.x_start) // 2

        elif direction == "left_very_middle":
            move = (self.x_end - self.x_start) // 2

        elif direction == "right_very_middle":
            move = -(self.x_end - self.x_start) // 4

        elif direction == "right_middle":
            move = -(self.x_end - self.x_start) // 2

        if not move is None:
            pyautogui.mouseDown(button='left')
            pyautogui.moveRel(move, 0, duration=1)
            pyautogui.mouseUp(button='left')
            pyautogui.moveRel(-move, 0, duration=1)
            return

        pyautogui.press(direction)

    def get_element_datetime(self):
        try:
            element = self.driver.find_element(By.XPATH, self.xpath["datetime"])
            image_data = base64.b64decode(self.driver.execute_script('return arguments[0].toDataURL().substring(21);', element))

            open('datetime/0.png', 'wb').write(image_data)
            
            text = image_to_text('datetime/0.png')
            
            if len(text) < len("26 Дек '23"):
                return False
            
            date = str_to_datatime(text)

            remove('datetime/0.png')

        except:
            return False
        
        return date

    def get_position(self):
        return ((self.x_start, self.y_start), (self.x_end, self.y_end))

    def set_position(self, position: tuple[tuple]):
        if not position:
            self.x_start, self.y_start = pyautogui.position()
            print(f"{self.x_start=} {self.y_start=}")
            input("Chekpoint #3")
            time.sleep(3) 
            self.x_end, self.y_end = pyautogui.position()
            print(f"{self.x_end=} {self.y_end=}")
            pyautogui.moveRel(-(self.x_end - self.x_start), 0, duration=1)
        else:
            self.x_start, self.y_start = position[0]
            self.x_end, self.y_end = position[1]

    def searh_datetime(self, datetime: datetime, position: tuple[tuple] = ()):
        if self.x_start is None:
            self.set_position(position)

        print(f"[INFO] START Searh - {datetime}")
        try:
            while True:
                
                date = self.get_element_datetime()

                if not date:
                    continue

                if datetime == date:
                    print(f"[INFO] END Searh")
                    return True

                if datetime < date:
                    if (datetime - date).days > -10:
                        direction = "left_middle"
                    elif (datetime - date).days > -1:
                        direction = "left_very_middle"
                    elif (datetime - date).days < 0:
                        direction = "left_fast"
                    else:
                        direction = "left"

                elif datetime > date:
                    if (datetime - date).days > 20:
                        direction = "right_middle"
                    elif (datetime - date).days > 1:
                        direction = "right_very_middle"
                    else:
                        direction = "right"

                self.move_cursor(direction)
                time.sleep(self.tick)
        except:
            return False

    
    def generate_for_df(self, df: pd.DataFrame):
        data = pd.DataFrame(columns=self.xpath.keys())
        temp_print_col_nan = 0
        temp_colnan = 0
        df = df.sort_values('datetime', ignore_index=True, ascending=False)
        
        self.switch_frame()
        
        if self.save:
            self.create_launch_dir()

        try:
            for date in df["datetime"]:
                if self.searh_datetime(date):
                    data_d = self.get_element()
                    print(len(data)+1, data_d)
                    data.loc[len(data)] = data_d
                    temp_colnan += 1
                else:
                    print(date, "-Not Found!")
                temp_print_col_nan += 1
                if temp_print_col_nan == 10:
                    col_nan = len(df) - temp_colnan
                    print(f"[INFO] {col_nan=}")
                    temp_print_col_nan = 0


            data["datetime"] = pd.to_datetime(data['datetime'])
        
        finally:
            self.end_web()
            return save_data(data) if self.save else data
        
    
    def create_launch_dir(self):
        mkdir(self.path_launch)
        chdir(self.path_launch)
        mkdir("datetime")

    def remove_launch_dir(self):
        chdir(self.main_dir)
        rmtree(self.path_launch)
