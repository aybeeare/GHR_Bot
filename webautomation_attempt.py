from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import webbrowser
import datetime
import time


def find_current_nyse_stocks():
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(options=options) # download from https://sites.google.com/chromium.org/driver/ and place in same directory as python script
    driver.get("http://www.nasdaq.com/market-activity/stocks/screener")
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//button[@id='onetrust-accept-btn-handler']"))).click()
    act = ActionChains(driver)
    act.click(driver.find_elements(By.XPATH, "//input[@id='radioItemNYSE']//a[@class='radioCircle']")).click(driver.find_elements(By.XPATH, "//a[@class='nasdaq-screener__form-button--download ns-download-1']")).perform()