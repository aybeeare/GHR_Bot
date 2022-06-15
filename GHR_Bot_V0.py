# GHR Bot will execute the investing stategy of Gerald Harris Rosen based on his book "A New Science of Stock Market Investing, 
# A Physicist Takes on Wall Street", coded by his grandson, Aaron Belkin-Rosen.

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
    print(tick_list)

# Generating P-V Graphs for Bearish/Bullish Classification
def generate_price_volume_data():

    for ticker_index in range(len(tick_list) - 1):
        # Connect to API using token in config file
        try:  # If data pull successful, generate weekly PV graph  
            #print(DailyJson['historical'])  
            days = 60
            API_Key_Daily = 'https://financialmodelingprep.com/api/v3/historical-price-full/' + str(tick_list[ticker_index]) + '?timeseries=' \
            + str(days) + '&type=ema&apikey=' + TOKEN
            # Pull Daily Price/Volume Data from API
            req = urllib.request.Request(API_Key_Daily) # instantiate request
            response = urllib.request.urlopen(req) # send request to API
            res_body = response.read().decode('utf-8') # read response
            DailyJson = json.loads(res_body) # transform response into json (data is a dictionary containing list of dictionaries)


            cum_vol = 0
            current_month_high = 0
            month_high = 0
            current_month_high_vol = 0
            month_high_vol = 0
            current_month_low = 0
            month_low = 0
            current_month_low_vol = 0
            month_low_vol = 0
            month_tracked = 0
            cum_vol_points = list()
            price_points = list()
            
            count = 0
            for i in range((len(DailyJson['historical']) -1), 0, -1): # iterating through list of dictionaries from past to future
                print(DailyJson['historical'][i])
                count += 1
                if month_tracked == 0:
                    if count == 1: # Find close price of first day in record
                        close_price = float(DailyJson['historical'][i]['close'])
                        current_month_low = float(DailyJson['historical'][i]['close']) # set current month low != 0
                        price_points.append(close_price)
                        cum_vol_points.append(cum_vol)
                    else: 
                        cum_vol += int(DailyJson['historical'][i]['volume'])
                        day_high = float(DailyJson['historical'][i]['high'])
                        day_low = float(DailyJson['historical'][i]['low'])

                        if day_high > current_month_high:
                            current_month_high = day_high
                            current_month_high_vol = cum_vol

                        if day_low < current_month_low:
                            current_month_low = day_low
                            current_month_low_vol = cum_vol

                        if count % 20 == 0: # once div by 20, 4 weeks passed, find month high, low and corresp. vols at each point
                            month_high = current_month_high
                            month_high_vol = current_month_high_vol
                            month_low = current_month_low
                            month_low_vol = current_month_low_vol
                            #print('Month High: ', tick_list[ticker_index], month_high)
                            #print('Month Low: ', tick_list[ticker_index], month_low)

                            if month_high_vol < month_low_vol: # find delta V selloff, if not, go back in time until one is found...
                                price_points.append(month_high)
                                cum_vol_points.append(month_high_vol)
                                price_points.append(month_low)
                                cum_vol_points.append(month_low_vol)
                                month_tracked = 1

                            else: # if month low comes first, ignore month high
                                break
                    
                    continue # move to next iteration until first month tracked is done

                if count % 2: # Every 2 days collect data for PV graph formation after tracking for 1 month
                    close_price = float(DailyJson['historical'][i]['close'])
                    cum_vol += int(DailyJson['historical'][i]['volume'])
                    price_points.append(close_price)
                    cum_vol_points.append(cum_vol)

            print(tick_list[ticker_index], price_points)
            print(tick_list[ticker_index], cum_vol_points)
                        

                #print(ticker_index, reversed(DailyJson['historical'][i]['date']), reversed(DailyJson['historical'][i]['close']), reversed(DailyJson['historical'][i]['high']), reversed(DailyJson['historical'][i]['low'])) 
        except:
            print('Failed: ', tick_list[ticker_index])
        
    print(tick_list)

        
# This was an initial debugging version, now refer to the main python script...
# print(len(close))
# print(len(vol))
# plt.plot(vol, close)
# plt.title('Microsoft P-V')
# plt.xlabel('Volume (millions of shares)')
# plt.ylabel('Price ($/share)')
# plt.show()
find_sib_stocks()
generate_price_volume_data()


# Machine Learning Model Training
