# GHR Bot will execute the investing stategy of Gerald Harris Rosen based on his book "A New Science of Stock Market Investing, 
# A Physicist Takes on Wall Street", coded by his grandson, Aaron Belkin-Rosen.

from tkinter.tix import DisplayStyle
import yaml
import pandas as pd
import matplotlib.pyplot as plt 
import webbrowser
import datetime
import csv
import re
import sys
import time
import json
import urllib.request

# Global variables


# Read API Keys from Config File
with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
TOKEN = cfg["TOKEN"]

# Filter NYSE Stocks and CIK
def nyse_filter():
    nyse_csv = pd.read_csv('nyse_stocks.csv') # download from https://www.nasdaq.com/market-activity/stocks/screener 
    # and save in same directory as python script
    nyse_list = nyse_csv['Symbol'].tolist()
    nyse_list_clean = list()
    # finds a ^ character in stock symbol and removes it from list
    for i in range(len(nyse_list)):
        if nyse_list[i].find('^') != -1: # contains ^
            continue
        else:
            nyse_list_clean.append(nyse_list[i]) 

# Find stocks with significant insider buying (transactions > $100,000)
def find_sib_stocks():
    API_KEY_INSIDER_TRADING = cfg["API_KEY_INSIDER_TRADING"]
    API_Key_Insider = API_KEY_INSIDER_TRADING + TOKEN

    # Pull Insider Trading Data from API
    req = urllib.request.Request(API_Key_Insider) # instantiate request
    response = urllib.request.urlopen(req) # send request to API
    res_body = response.read().decode('utf-8') # read response
    filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)
    
    # Find sib stocks (includes all stock market indices (NYSE, NASDAQ, ...))
    # and write to csv file for personal historical documentation
    file_path = 'C:\\Users\\abelk\\OneDrive\\Desktop\\GHR_Bot\\ghr_bot_sib_record.csv'
    f = open(file_path, 'a', newline = '')

    global tick_list 
    tick_list = list() # no reps in ticks
    sib_dict_count = dict() # keep track of how many significant insider buys occurred per stock

    for dics in filingsJson:
        tick = dics.get('symbol')
        trans_type = dics.get('acquistionOrDisposition') # 'A' or 'D'
        trans_amount = int(dics.get('price'))*int(dics.get('securitiesTransacted'))
        trans_date = dics.get('transactionDate')
        if trans_type == 'A' and trans_amount > 100000:
            if tick not in tick_list:
                new_row = [tick, trans_type, trans_amount, trans_date] 
                writer_obj = csv.writer(f)
                df = pd.read_csv(file_path)
                if (trans_date not in df['Trans Date'].values.tolist() or tick not in df['Symbol'].values.tolist()):
                    writer_obj.writerow(new_row)     
                tick_list.append(tick)
            if tick not in sib_dict_count:
                sib_dict_count[tick] = 1 
            else:
                sib_dict_count[tick] += 1
            
    f.close()
    tick_list[1] = 'NOG'
    tick_list[2] = 'GE'

# Function to pull JSON data from API
def api_pull(days):
    try: 
        API_Key_Daily = 'https://financialmodelingprep.com/api/v3/historical-price-full/' + str(tick_list[index]) + '?timeseries=' \
        + str(days) + '&type=ema&apikey=' + TOKEN
        # Pull Daily Price/Volume Data from API
        req = urllib.request.Request(API_Key_Daily) # instantiate request
        response = urllib.request.urlopen(req) # send request to API
        res_body = response.read().decode('utf-8') # read response
        global DailyJson
        DailyJson = json.loads(res_body) # transform response into json (data is a dictionary containing list of dictionaries)
        print('Successful API Pull: ', tick_list[index])
        return 0
    except:
        print('Failed API Pull: ', tick_list[index])
        return -1

def pv_attempt():  
    cum_vol = 0
    current_month_high = 0
    month_high = 0
    current_month_high_vol = 0
    month_high_vol = 0
    current_month_low = 0
    month_low = 0
    current_month_low_vol = 0
    month_low_vol = 0
    delta_vee = False
    cum_vol_points = list()
    price_points = list()
    global api_status
    global div
    global days
    global pv_attempts
    
    count = 0
    try:
        for i in range((len(DailyJson['historical']) -1), 0, -1): # iterating through list of dictionaries from past to future
            #print(DailyJson['historical'][i])
            count += 1
            if not delta_vee:
                if count == 1: # Find close price of first day in record
                    close_price = float(DailyJson['historical'][i]['close'])
                    current_month_low = float(DailyJson['historical'][i]['close']) 
                    current_month_high = float(DailyJson['historical'][i]['close']) 
                    price_points.append(close_price)
                    cum_vol_points.append(cum_vol)
                else:
                    try: 
                        cum_vol += int(DailyJson['historical'][i]['volume'])
                        day_close = float(DailyJson['historical'][i]['close'])
                    
                    except: 
                        print('Broke, could not get stock data for: ', tick_list[index])
                        break

                    if day_close > current_month_high:
                        current_month_high = day_close
                        current_month_high_vol = cum_vol

                    if day_close < current_month_low:
                        current_month_low = day_close
                        current_month_low_vol = cum_vol

                    if count % div == 0: # once div by 20, month passed, 40, 2 months... find high, low and corresp. vols at each point
                        month_high = current_month_high
                        month_high_vol = current_month_high_vol
                        month_low = current_month_low
                        month_low_vol = current_month_low_vol
                        print('Month High: ', tick_list[index], month_high_vol)
                        print('Month Low: ', tick_list[index], month_low_vol)

                        if month_high_vol < month_low_vol: # find delta V selloff, if not, go back in time until one is found...
                            price_points.append(month_high)
                            cum_vol_points.append(month_high_vol)
                            price_points.append(month_low)
                            cum_vol_points.append(month_low_vol)
                            delta_vee = True # sell off volume found!

                        else: # do another api pull going back farther and attempt to generate pv again
                            days = days + 20
                            div = div + 20
                            print('Current Days: ', days)
                            print('Current Div: ', div)
                            api_status = api_pull(days) # new data stored in DailyJson after pull?
                            print('TRYING AGAIN, API Status: ',api_status)
                            if api_status == 0:
                                pv_attempt()
                                pv_attempts = pv_attempts + 1
                                print(tick_list[index], 'PV Attempts: ', pv_attempts)
                                continue
                            else:
                                print(tick_list[index], 'Failed to generate PV with delta vee')
                                sys.exit()
                    else:
                        continue
                
            if count % 2: # Every 2 days collect data for PV graph formation after tracking for 1 month
                close_price = float(DailyJson['historical'][i]['close'])
                cum_vol += int(DailyJson['historical'][i]['volume'])
                price_points.append(close_price)
                cum_vol_points.append(cum_vol)

        print('Price List: ', tick_list[index], price_points)
        print('Vol List: ', tick_list[index], cum_vol_points)
        return 0

    except: 
        print(tick_list[index], 'was skipped')
        return -1
         # move to next iteration until first month tracked is done

        

# Generating P-V Graphs for Bearish/Bullish Classification

def generate_price_volume_data():       
    for ticker_index in range(len(tick_list) -1):      
        global index
        index = ticker_index
        print('Tick List: ', tick_list)
        print('Index: ', index)
        global days
        days = 60
        global div
        div = 20
        global pv_attempts
        pv_attempts = 0
        api_status = api_pull(days)
        print('did api pull')
        print(tick_list[index], 'API Status', api_status)
        pv_attempt_status = pv_attempt()
        print(tick_list[index], 'PV Status: ', pv_attempt_status)
                

        
# print(hist)
# print(list(hist.columns))

# print(len(close))
# print(len(vol))
# plt.plot(vol, close)
# plt.title('Microsoft P-V')
# plt.xlabel('Volume (millions of shares)')
# plt.ylabel('Price ($/share)')
# plt.show()
#find_sib_stocks()
#tick_list = list()
#tick_list.append('APPN')
find_sib_stocks()
generate_price_volume_data()


# Machine Learning Model Training

# Execute
#find_current_nyse_stocks()
