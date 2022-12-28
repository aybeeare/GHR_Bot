# GHR Bot will execute the investing stategy of Gerald Harris Rosen based on his book "A New Science of Stock Market Investing, 
# A Physicist Takes on Wall Street", coded by his grandson, Aaron Belkin-Rosen.

import platform
import sys
import os
import csv
import datetime
import json
import urllib.request
import yaml
import pandas as pd
import matplotlib.pyplot as plt

with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
global TOKEN
TOKEN = cfg["TOKEN"]

def setup(): # most likely not going to use!
    welcome = input("Welcome to GHR BOT! Is this your first time using it? (Y/N)")
    if welcome == 'Y':
        os_type = platform.system() # returns 'Windows', 'Linux', or 'Darwin' (mac is Darwin lol)
        if os_type == 'Windows':
            print('INSTALLING PYTHON LIBRARIES FOR WINDOWS...')
            os.system('python -m pip install --upgrade pip')
            os.system('pip install PyYAML')
            os.system('pip install matplotlib')
            os.system('pip install pandas')
            print('DONE INSTALLING PYTHON LIBRARIES!')

            # import libraries after install

            global yaml
            import yaml
            global pd
            import pandas as pd
            global plt
            import matplotlib.pyplot as plt

        elif os_type == 'Linux' | 'Darwin':
            print('INSTALLING PYTHON LIBRARIES FOR LINUX...')
            os.system('sudo apt-get update')
            os.system('sudo apt-get -y install python3-pip')
            os.system('pip3 install PyYAML')
            os.system('pip3 install matplotlib')
            os.system('pip3 install pandas')
            print('DONE INSTALLING PYTHON LIBRARIES!')

            # import libraries after install

            global yaml
            import yaml
            global pd
            import pandas as pd
            global plt
            import matplotlib.pyplot as plt

        else:
            setup()
    else:
        global yaml
        import yaml
        global pd
        import pandas as pd
        global plt
        import matplotlib.pyplot as plt

def list_all_stocks():
    global stocks_tup
    list_stocks = 'https://financialmodelingprep.com/api/v3/stock/list?apikey=' + TOKEN
    req = urllib.request.Request(list_stocks) # instantiate request
    response = urllib.request.urlopen(req) # send request to API
    res_body = response.read().decode('utf-8') # read response
    filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)
    stocks = []
    for dic in filingsJson:
        # If keyboard interrupt, break and return stocks_tup
        if (dic.get('exchangeShortName') == 'NASDAQ' or dic.get('exchangeShortName') == 'NYSE' or dic.get('exchangeShortName') == 'AMEX') and dic.get('type') == 'stock':
            stocks.append(dic.get('symbol')) 
    stocks_tup = tuple(stocks)

# Find all sibs for all stocks within the past 2 months
def recent_sibs():
    # Find current date and time to check how recent and only run this code periodically and allow for keyboard interrupts
    date = str(datetime.date.today()).split('-')
    global yr, mth
    yr, mth, day = date
    count = 0

    # Instantiate empty dataframe containing symbol, trans_type, trans_amount, trans_date, buyer
    COLUMN_NAMES = ['Symbol','Insider', 'Ins-Count', 'Ins-Date', 'Ins-Amt', 'Senator', 'Sen-Date', 'Sen-Amt','Representative', 'Rep-Date', 'Rep-Amt', 'Press', 'Bargain', 'PV Trend', 'Buy Strength']
    
    df = pd.DataFrame(columns=COLUMN_NAMES)
    symbol_list = []
    trans_list = []
    reg_insider_list = []

    API_KEY_INSIDER_BUYING = cfg["API_KEY_INSIDER_TICK"]

    # Conditional for how often to run this code (should be done periodically while GUI is IDLE/Closed and Hidden CSV should be generated)

    for tick in stocks_tup:
        # if count >= 200: # Run a certain number of times, takes a whle :)
        #     break
        count += 1

        #print('Tick: ', tick)
        API_Key_Insider = API_KEY_INSIDER_BUYING + str(tick) + '&page=0&apikey=' + TOKEN
        #print('API KEY: ', API_Key_Insider)

        # Pull Insider Trading Data from API
        req = urllib.request.Request(API_Key_Insider) # instantiate request
        response = urllib.request.urlopen(req) # send request to API
        res_body = response.read().decode('utf-8') # read response
        filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)
        
        #print(filingsJson)

        # Compare significant insider sells with portfolio holdings, if match, alert...
        for dics in filingsJson:
            #print(dics)
            try:
                tick = dics.get('symbol')
                trans_type = dics.get('transactionType')[0] # 'S-Sale' or 'P-Purchase'
                trans_amount =  (float(dics.get('price')))*(float(dics.get('securitiesTransacted')))
                trans_date = dics.get('transactionDate')
                year, month, day = trans_date.split('-')
                insider = dics.get('reportingName')
                security = dics.get('securityName')
            
            except:
                print('Continued!', count)
                continue

            if trans_type == 'P' and trans_amount > 300000 and 'Common' in security and (((int(mth) - int(month) <= 2) and yr == year) or (int(yr) - int(year) == 1 and ((int(month) - int(mth)) >= 10))):
                if tick not in symbol_list or trans_date not in trans_list or insider not in reg_insider_list:
                    df.loc[len(df)] = [tick, insider, None, trans_date, trans_amount, None, None, None, None, None, None, None, None, None, None] # append new row to end of df
                    symbol_list.append(tick)
                    trans_list.append(trans_date)
                    reg_insider_list.append(insider)
        
    # Write df to CSV
    print(df)
    df.to_csv('ghr_bot_sib_record2.csv')

# Find stocks with significant insider buying (transactions > $100,000)
def current_sib_stocks():
    
    # Read ghr_bot_sib_record2.csv as df and append new sib stocks
    
    df = pd.read_csv('ghr_bot_sib_record2.csv')
    symbol_list = df['Symbol'].tolist()
    trans_list = df['Ins Date'].tolist()
    reg_insider_list = df['Insider'].tolist()

    API_KEY_INSIDER_TRADING = cfg["API_KEY_INSIDER_TRADING"]
    API_Key_Insider = API_KEY_INSIDER_TRADING + TOKEN

    # Pull Insider Trading Data from API
    req = urllib.request.Request(API_Key_Insider) # instantiate request
    response = urllib.request.urlopen(req) # send request to API
    res_body = response.read().decode('utf-8') # read response
    filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)

    for dics in filingsJson:
        tick = dics.get('symbol')
        trans_type = dics.get('transactionType')[0] # 'S-Sale' or 'P-Purchase'
        trans_amount = int(dics.get('price'))*int(dics.get('securitiesTransacted'))
        trans_date = dics.get('transactionDate')
        year, month, day = trans_date.split('-')
        insider = dics.get('reportingName')
        security = dics.get('securityName')
        
        if trans_type == 'P' and trans_amount > 300000 and 'Common' in security and (((int(mth) - int(month) <= 2) and yr == year) or (int(yr) - int(year) == 1 and ((int(month) - int(mth)) >= 10))):
            if tick not in symbol_list or trans_date not in trans_list or insider not in reg_insider_list:
                df.loc[len(df)] = [tick, insider, None, trans_date, trans_amount, None, None, None, None, None, None, None, None, None, None]# append new row to end of df
                symbol_list.append(tick)
                trans_list.append(trans_date)
                reg_insider_list.append(insider)
    
    df.to_csv('ghr_bot_sib_record2.csv')

# Cleaned up df to go inside rec algo 
def df_cleanup():

    # Read df from csv and iterate through symbols
    df = pd.read_csv('ghr_bot_sib_record2.csv')
    symbol_list = tuple(df['Symbol'].tolist())
    insider = tuple(df['Insider'].tolist())
    trans_amt = tuple(df['Ins-Amt'].tolist())
    trans_date = tuple(df['Ins-Date'].tolist())
    cleanup_dict = {}
    ins_list = []
    date_list = []

    for sym, ins, amt, date in zip(symbol_list, insider, trans_amt, trans_date):
        #print('Sym: ', 'Insider: ', 'Amount', sym, ins, date)
        if sym not in cleanup_dict.keys():
            count = 1
            cum_amt = amt
            cleanup_dict[sym] = [[ins], count, [date], cum_amt]
                
        else: # if sym is in cleanup_dict already
            if (ins not in cleanup_dict[sym][0] or date not in cleanup_dict[sym][2]): # Count number of unique buys
                
                cleanup_dict[sym][0].append(ins)
                cleanup_dict[sym][1] += 1
                cleanup_dict[sym][2].append(date)
                cleanup_dict[sym][3] += amt

    # Find most recent date, corresponding insider, and average sib size to store in cleaned up df
    for sym in cleanup_dict.keys():
        most_rec = 0
        most_rec_idx = 0

        for idx in range(len(cleanup_dict[sym][2])):
            date = int(cleanup_dict[sym][2][idx].replace('-',''))

            if date > most_rec:
                most_rec = date
                most_rec_idx = idx

        cleanup_dict[sym][0] = cleanup_dict[sym][0][most_rec_idx] # select corresponding insider w/ most recent date
        cleanup_dict[sym][2] = cleanup_dict[sym][2][most_rec_idx]
        cleanup_dict[sym][3] = int(cleanup_dict[sym][3] / cleanup_dict[sym][1]) # avg trans = cum/count

    # Instantiate empty dataframe containing symbol, trans_type, trans_amount, trans_date, buyer

    COLUMN_NAMES = ['Insider', 'Ins-Count', 'Ins-Date', 'Ins-Amt'] # ... 'Senator', 'Sen-Date', 'Sen-Amt','Representative', 'Rep-Date', 'Rep-Amt', 'Press', 'Bargain', 'PV Trend', 'Buy Strength']
    

    df_clean = pd.DataFrame.from_dict(cleanup_dict, orient= 'index', columns=COLUMN_NAMES)
    df_clean.to_csv('ghr_bot_sib_record2_cp.csv')
            

# Extend insiders to track members of senate and house and consolidate in dataframe
def sib_extend():

    # Read ghr_bot_sib_record2.csv as df and append new sib stocks
    df = pd.read_csv('ghr_bot_sib_record2_cp.csv')
    symbol_list = list(df.index.values)

    # Find current date
    senate_dict = {}
    house_dict = {}
    date = str(datetime.date.today()).split('-')
    yr, mth, day = date

    # Get Json for House and Senate
    url_house = 'https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json'
    url_senate = 'https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json'
    req_house = urllib.request.Request(url_house) # instantiate request
    response_house = urllib.request.urlopen(req_house) # send request to API
    res_body_house = response_house.read().decode('utf-8') # gives a json string
    json_house = json.loads(res_body_house)
    req_senate = urllib.request.Request(url_senate) # instantiate request
    response_senate = urllib.request.urlopen(req_senate) # send request to API
    res_body_senate = response_senate.read().decode('utf-8') # gives a json string
    json_senate = json.loads(res_body_senate)
    count_senate = 0
    count_house = 0
    
    # Add all house buys to sib dataframe
    for x in json_house: 
        trans_date = x['transaction_date'].split('-')
        year, month, day = trans_date

        if (((int(mth) - int(month) <= 12) and yr == year) or (int(yr) - int(year) == 1 and (int(month) > int(mth)))) and ('purchase' in x['type']): # logic for less than one year
            # Check if ticker already in house dict, if not, instantiate, if it is, append to list
            if x['ticker'] not in house_dict.keys():
                house_dict[x['ticker']] = [[x['representative']], [x['transaction_date']], [x['amount']]]
            else:
                house_dict[x['ticker']][0].append(x['representative'])
                house_dict[x['ticker']][1].append(x['transaction_date'])
                house_dict[x['ticker']][2].append(x['amount'])         
            
    # Prints all the senate senate buys
    for x in json_senate:
        trans_date = x['transaction_date'].split('/')
        month, day, year = trans_date
        #if (((int(mth) - int(month) <= 12) and yr == year) or (int(yr) - int(year) == 1 and (int(month) > int(mth)))) and ('Purchase' in x['type']): # logic for less than one year
        if (((int(mth) - int(month) <= 3) and yr == year) or (int(yr) - int(year) == 1 and ((int(month) - int(mth)) >= 9))) and ('Purchase' in x['type']): # logic for less than 3 months
            # Check if ticker already in senate dict, if not, instantiate, if it is, append to list
            if x['ticker'] not in senate_dict.keys():
                senate_dict[x['ticker']] = [[x['senator']], [x['transaction_date']], [x['amount']]]  
            else:
                senate_dict[x['ticker']][0].append(x['senator'])
                senate_dict[x['ticker']][1].append(x['transaction_date'])
                senate_dict[x['ticker']][2].append(x['amount'])
                
            count_senate += 1

    # Add senate and house buys to cleaned data frame. First check if tick already in df, if it is, edit key, if not, add. 
    for senate_buy, house_buy in zip(senate_dict.keys(), house_dict.keys()):
        print(house_buy)

    # Combine dictionaries and consolidate 
    # print('################ HOUSE ################## \n')
    # print(house_dict)
    # print('################  ################## \n')
    # print(senate_dict)
    

# Find stocks with significant insider buying (transactions > $100,000)
def find_sis_of_portfolio():
    API_KEY_INSIDER_TRADING = cfg["API_KEY_INSIDER_TRADING"]
    API_Key_Insider = API_KEY_INSIDER_TRADING + TOKEN

    # Pull Insider Trading Data from API
    req = urllib.request.Request(API_Key_Insider) # instantiate request
    response = urllib.request.urlopen(req) # send request to API
    res_body = response.read().decode('utf-8') # read response
    filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)
    
    # Open sib record and get list of symbols
    current_dir = os.getcwd()
    file_path = current_dir + '\my_portfolio.csv'

    try:
        f = open(file_path, 'a+')
    except:
        print('ERROR: Please close excel spreadsheet "my_portfolio.csv" and run program again\n\n')

    portfolio_list = ['NRDY', 'WAL', 'CFLT', 'PARA', 'DNB', 'OXY', 'NILE', 'RVMD', 'SUP', 'TDW', 'ET']

    # Compare significant insider sells with portfolio holdings, if match, alert...
    for dics in filingsJson:
        tick = dics.get('symbol')
        trans_type = dics.get('transactionType')[0] # 'S-Sale' or 'P-Purchase'
        trans_amount = int(dics.get('price'))*int(dics.get('securitiesTransacted'))
        trans_date = dics.get('transactionDate')
        seller = dics.get('reportingName')
        if trans_type == 'S' and trans_amount > 100000 and tick in portfolio_list:
            df = pd.read_csv(file_path)
            if (trans_date not in df['Trans Date'].values.tolist() or tick not in df['Symbol'].values.tolist() or seller not in df['Seller']):
                new_row = [tick, trans_type, trans_amount, trans_date, seller] 
                writer_obj = csv.writer(f)
                writer_obj.writerow(new_row)     
                # send notification of insider sell with all details!       
    f.close()

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
    global price_points
    global cum_vol_points
    cum_vol_points = list()
    price_points = list()
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
                        print('Month High: ', tick_list[index], month_high)
                        print('Month Low: ', tick_list[index], month_low)

                        if month_high_vol < month_low_vol: # find delta V selloff, if not, go back in time until one is found...
                            price_points.append(month_high)
                            cum_vol_points.append(month_high_vol)
                            price_points.append(month_low)
                            cum_vol_points.append(month_low_vol)
                            delta_vee = True # sell off volume found!
                            continue

                        else: # do another api pull going back farther and attempt to generate pv again
                            print('Doing another API Pull For: ', tick_list[index])
                            days = days + 10 # this works way better with starting at 40 days and div = 20...
                            div = div + 5
                            return -2
                            
                    else:
                        continue
                
            # Once delta vee found, pull every day
            close_price = float(DailyJson['historical'][i]['close'])
            cum_vol += int(DailyJson['historical'][i]['volume'])
            price_points.append(close_price)
            cum_vol_points.append(cum_vol)
    
        cum_vol_points = [x * 0.000001 for x in cum_vol_points] # convert vol to millions of shares and round to 2 decimals
        cum_vol_points = [round(x,3) for x in cum_vol_points]
        del cum_vol_points[1] 
        del price_points[1] 
        print('Price List: ', tick_list[index], price_points) # why does it jump here and not do the count % 2 on each new api pull?
        print('Vol List: ', tick_list[index], cum_vol_points)
        return 0

    except: 
        print(tick_list[index], 'was skipped')
        return -1
         # move to next iteration until first month tracked is done

def classify_pv_bearish_bullish(vol_inputs, price_inputs): 
    PERCENT_TOLERANCE = 0.06
    classification = 'Bearish'
    print('Currently PV Classifying... ', tick_list[index])
    cum_vol_inputs = vol_inputs # vol_inputs and price_inputs generated from pv attempt 
    price_point_inputs = price_inputs
    counts = 0
    indices_list = list()
    print(tick_list[index], 'Cum_Vol_Points: ', cum_vol_inputs)
    print(tick_list[index], 'Price_Points: ', price_point_inputs)
    try: # try to get delta vee points from data, if not, break it
        delta_vee = cum_vol_inputs[2] - cum_vol_inputs[1]
        print(tick_list[index],'Delta Vee: ', delta_vee) 
        delta_vee_vol_start = cum_vol_inputs[2]
        delta_vee_price_start = price_point_inputs[2]
    except:
        print('Failed to generate delta vee for: ', tick_list[index])
        return -1
    
    # doesnt account for multiple cycles in one graph!!!
    print('Length Cum_Vol_Inputs: ',len(cum_vol_inputs))
    for i in range(len(cum_vol_inputs)):
        counts += 1
        if counts > 3: # look at points starting at end of delta v and check volume criteria for classifiation switch
            indices_list.append(i)
            print('Current Cum_Vol: ', cum_vol_inputs[i])
            print('Current Price Point: ', price_point_inputs[i])
            if (cum_vol_inputs[i] > (delta_vee_vol_start + delta_vee)) and classification == 'Bearish': # volume criteria for classification switch passed
                classified = False 

                for idx in indices_list:

                    if cum_vol_inputs[idx] - cum_vol_inputs[idx -1] >= delta_vee: # for case where next cum vol is bigger than previous by delta vee.
                        if price_point_inputs[idx] > price_point_inputs[idx -1]:
                            classification = 'Bullish'
                        else:
                            classification = 'Bearish'
                        delta_vee_vol_start = cum_vol_inputs[idx]
                        delta_vee_price_start = price_point_inputs[idx]
                        indices_list.clear()
                        break

                    if (price_point_inputs[idx] < (delta_vee_price_start - delta_vee_price_start*PERCENT_TOLERANCE)): # If point below start by good amount, keep bearish
                        delta_vee_vol_start = cum_vol_inputs[indices_list[-1]] # vol and price at last indice of list are start of next delta vee cycle 
                        delta_vee_price_start = price_point_inputs[indices_list[-1]]
                        classified = True
                        
                    if idx == indices_list[-1] and classified == False: # If last index in indices reached and still false, all were above min!
                        classification = 'Bullish'
                        delta_vee_vol_start = cum_vol_inputs[indices_list[-1]]
                        delta_vee_price_start = price_point_inputs[indices_list[-1]]
                
                indices_list.clear()
                        
            if (cum_vol_inputs[i] > (delta_vee_vol_start + delta_vee)) and classification == 'Bullish': # volume criteria for classification switch passed
                classified = False 

                for idx in indices_list:
                    
                    if cum_vol_inputs[idx] - cum_vol_inputs[idx -1] >= delta_vee: # for case where next cum vol is bigger than previous by delta vee.
                        if price_point_inputs[idx] < price_point_inputs[idx -1]:
                            classification = 'Bearish'
                        else:
                            classification = 'Bullish'
                        delta_vee_vol_start = cum_vol_inputs[idx]
                        delta_vee_price_start = price_point_inputs[idx]
                        indices_list.clear()
                        break

                    if price_point_inputs[idx] > (delta_vee_price_start + delta_vee_price_start*PERCENT_TOLERANCE): # If a point is above start by good amt, keep bullish.
                        delta_vee_vol_start = cum_vol_inputs[indices_list[-1]] # vol and price at last indice of list are start of next delta vee cycle 
                        delta_vee_price_start = price_point_inputs[indices_list[-1]]
                        classified = True
                             
                    if idx == indices_list[-1] and classified == False: # If last index in indices reached and still false, all were above min!
                        classification = 'Bearish'
                        delta_vee_vol_start = cum_vol_inputs[indices_list[-1]]
                        delta_vee_price_start = price_point_inputs[indices_list[-1]]
                
                indices_list.clear()
            print(tick_list[index], classification)

    print('Classification is: ', classification)         
    classification_dict[tick_list[index]] = classification

        
# Generating P-V Graphs for Bearish/Bullish Classification
def generate_pv_and_plot(input_list):
    global tick_list
    tick_list = input_list      
    for ticker_index in range(len(tick_list) -1):      
        global index
        index = ticker_index
        global days
        days = 40
        global div
        div = 20
        global pv_attempts
        pv_attempts = 0
        api_status = api_pull(days)
        print(tick_list[index], 'API Status', api_status)
        pv_attempt_status = pv_attempt()
        while pv_attempt_status == -2:
            api_pull(days)
            pv_attempts += 1
            print(tick_list[index], ' PV attempts: ', pv_attempts)
            pv_stat = pv_attempt()
            if pv_stat == 0:
                print('Finally PV Works!: ', tick_list[index])
                break
           
        print(tick_list[index], 'PV Status: ', pv_attempt_status)
        # Plot PV for Stock:
        try:
            plt.title('PV for Significant Insider Buying Stock: '+ str(tick_list[index])) 
            plt.plot(cum_vol_points, price_points)
            plt.xlabel('Volume (millions of shares)')
            plt.ylabel('Price ($/share)')
            plt.show()
        except:
            print('Could not plot: ', tick_list[index])
            continue
    
        if ticker_index == range(len(tick_list) -1)[-1]:
            print('\n\n\n\n\n\n\n\n\nDONE!\nTHANK YOU FOR CHOOSING THE GHR BOT!\nYOURS,\nABR')

# Generate PV for every sib+ ticker throughout this program's history
def gen_sib_positive_pv_graphs():
    special_list = True # set to true if you want to put in your own tickers, if not, pulls from ghr_bot_sib_record
    sib_pos_csv = pd.read_csv('ghr_bot_sib_record.csv')  
    # and save in same directory as python script
    sib_pos_ticks = sib_pos_csv['Symbol'].tolist()
    sib_pos_ticks = [x for x in sib_pos_ticks if str(x) != 'nan']
    if special_list:
        sib_pos_ticks.clear()
        user_input = input("Enter Ticker(s) you would like to do PV analysis on (separate all inputs by commas and type 'DONE' and hit ENTER key when finished): ") 
        stocks_input = user_input.split(',')
        for stock in stocks_input:
            sib_pos_ticks.append(stock.strip())
        print(sib_pos_ticks)

        if sib_pos_ticks[-1] == 'DONE':
            generate_pv_and_plot(sib_pos_ticks)
              #sib_pos_ticks = ['SUP','ALDX', 'MVST'] # enter whichever tickers you want to look at...
        else:
            print('USER INPUT ERROR: Please type "DONE" and press "ENTER" key when finished inputting stocks of interest/n/n')
            setup()

    # API call by ticker to get insider trading by symbol https://financialmodelingprep.com/api/v4/insider-trading?symbol= + str(tick_list) + '&page=0'

# Generating P-V Graphs for Bearish/Bullish Classification
def generate_pv_and_classify(input_list):
    global tick_list
    tick_list = input_list
    global classification_dict
    classification_dict = dict()      
    for ticker_index in range(len(tick_list) -1):      
        global index
        index = ticker_index
        global days
        days = 40
        global div
        div = 20
        global pv_attempts
        pv_attempts = 0
        api_status = api_pull(days)
        print(tick_list[index], 'API Status', api_status)
        pv_attempt_status = pv_attempt()
        while pv_attempt_status == -2:
            api_pull(days)
            pv_attempts += 1
            print('PV Attempts: ', pv_attempts)            
            pv_stat = pv_attempt() 
            if pv_stat == 0:
                break
            if pv_attempts > 7:
                print('Could not generate PV for: ', tick_list[index])
                break

        classify_pv_bearish_bullish(cum_vol_points, price_points) 
    print(classification_dict)  

# returns a dataframe of ticks w/ insider buying, current pric (deal or not), pv trend, press release
# soon after insider buy? 
def gen_prospective_buys_df(): 
    print()

def gen_portfolio_management_df():
    pass

def recommendation_algo(): 
    pass

# Execute sequence of function calls to run program (First release package!)

# setup()
# list_all_stocks()
# recent_sibs()
#current_sib_stocks()
df_cleanup()
#sib_extend()
# gen_sib_positive_pv_graphs()


# Test sequence for classifier

test_list = ['NRDY', 'WAL', 'ET', 'PARA', 'DNB', 'OXY', 'NILE', 'RVMD', 'SUP', 'TDW', 'DONE']

# Insiders and politicians overlapping (insiders over last 2 months and politicians over last year)
#test_list = ['SAVA', 'CIVI', 'ECL', 'ET', 'NEE', 'KKR', 'SBUX', 'BX', 'INTC', 'DINO', 'XOM', 'ARQT', 'WBD', 'SCI', 'JEF', 'COIN', 'MTCH', 'MRVL' 'CAG', 'DONE']
#test_list = ['ET', 'SBUX', 'DINO', 'SCI', 'SNX', 'BRZE', 'FRSH', 'INSM', '',  'DONE'] # overlap...
#generate_pv_and_plot(test_list)
#generate_pv_and_classify(test_list)


