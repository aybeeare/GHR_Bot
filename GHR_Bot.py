# GHR Bot will execute the investing stategy of Gerald Harris Rosen based on his book "A New Science of Stock Market Investing, 
# A Physicist Takes on Wall Street", coded by his grandson, Aaron Belkin-Rosen.

import os
import csv
import json
import urllib.request

def setup():
    welcome = input("Welcome to GHR BOT! Is this your first time using it? (Y/N)")
    if welcome == 'Y':
        os_type = input("Is your PC running a Windows or Linux OS? (W/L)")
        if os_type == 'W':
            print('INSTALLING PYTHON LIBRARIES FOR WINDOWS...')
            os.system('python -m pip install --upgrade pip')
            os.system('pip install PyYAML')
            os.system('pip install matplotlib')
            os.system('pip install pandas')
            print('DONE INSTALLING PYTHON LIBRARIES!!')

            # import libraries after install

            global yaml
            import yaml
            global pd
            import pandas as pd
            global plt
            import matplotlib.pyplot as plt

        elif os_type == 'L':
            print('INSTALLING PYTHON LIBRARIES FOR LINUX...')
            os.system('sudo apt-get update')
            os.system('sudo apt-get -y install python3-pip')
            os.system('pip3 install PyYAML')
            os.system('pip3 install matplotlib')
            os.system('pip3 install pandas')
            print('DONE INSTALLING PYTHON LIBRARIES!!')

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
    
    # Find sib stocks (includes all stock market indices (NYSE, NASDAQ, ...)
    # and write to csv file for personal historical documentation
    current_dir = os.getcwd()
    file_path = current_dir + '\ghr_bot_sib_record.csv'
    f = open(file_path, 'a', newline = '')

    global tick_list 
    tick_list = list() # no reps in ticks
    global sib_dict_count
    sib_dict_count = dict() # keep track of how many significant insider buys occurred per stock

    for dics in filingsJson:
        tick = dics.get('symbol')
        trans_type = dics.get('acquistionOrDisposition') # 'A' or 'D'
        trans_amount = int(dics.get('price'))*int(dics.get('securitiesTransacted'))
        trans_date = dics.get('transactionDate')
        buyer = dics.get('reportingName')
        if trans_type == 'A' and trans_amount > 500000:
            if tick not in tick_list:
                new_row = [tick, trans_type, trans_amount, trans_date, buyer] 
                writer_obj = csv.writer(f)
                df = pd.read_csv(file_path)
                if (trans_date not in df['Trans Date'].values.tolist() or tick not in df['Symbol'].values.tolist() or buyer not in df['Buyer']):
                    writer_obj.writerow(new_row)     
                tick_list.append(tick)
            if tick not in sib_dict_count:
                sib_dict_count[tick] = 1 
            else:
                sib_dict_count[tick] += 1          
    f.close()

# Find stocks with significant insider buying (transactions > $100,000)
def find_sis_of_sibs():
    pass
    # going to be very similar to find_sib function

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
                            days = days + 20
                            div = div + 10
                            return -2
                            
                    else:
                        continue
                
            if count % 2: # Every 2 days collect data for PV graph formation after tracking for 1 month
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
    global classification_dict
    classification = 'Bearish'
    classification_dict = dict()
    print('Currently PV Classifying... ', tick_list[index])
    cum_vol_inputs = vol_inputs  
    price_point_inputs = price_inputs
    counts = 0
    indices_list = list()
    try: # try to get delta vee points from data, if not, break it
        delta_vee = cum_vol_inputs[2] - cum_vol_inputs[1] 
        delta_vee_vol_min = cum_vol_inputs[2]
        delta_vee_price_min = price_point_inputs[2]
    except:
        print('Failed to generate delta vee for: ', tick_list[index])
        return -1
    
    # doesnt account for multiple cycles in one graph!!!
    for i in range(len(cum_vol_inputs)):
        counts += 1
        if counts > 3: # look at points starting at end of delta v and check volume criteria for classifiation switch
            indices_list.append(i)
            if (cum_vol_inputs[i] > (delta_vee_vol_min + delta_vee)): # volume criteria for classification switch passed
                print('Indices List: ', indices_list) # indices after delta vee scanned for next eligible switch evaluation
                delta_vee_vol_min = cum_vol_inputs[i] # update starting vol for next classification switch cycle
                for idx in indices_list:
                    print('Current Delta Vee Index Is: ', idx)
                    print('Delta Vee Price List: ', price_point_inputs[(indices_list[0]):(indices_list[-1])])
                    print('Delta Vee Vol List: ', cum_vol_inputs[(indices_list[0]):(indices_list[-1])])
                    if price_point_inputs[idx] < delta_vee_price_min:
                        classification = 'Bearish'
                        absolute_min = delta_vee_price_min
                        if idx == indices_list[-1]: 
                            indices_list.clear()
                            delta_vee_price_min = absolute_min
                            break
                    if all(x > delta_vee_price_min for x in price_point_inputs[(indices_list[0]):(indices_list[-1])]):
                        classification = 'Bullish'
                        indices_list.clear()
                        break
    
    classification_dict.update({tick_list[index] : classification})

        
# Generating P-V Graphs for Bearish/Bullish Classification
def generate_price_volume_data(input_list):
    global tick_list
    tick_list = input_list      
    for ticker_index in range(len(tick_list) -1):      
        global index
        index = ticker_index
        global days
        days = 60
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
        classify_pv_bearish_bullish(cum_vol_points, price_points)
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
            generate_price_volume_data(sib_pos_ticks)
              #sib_pos_ticks = ['SUP','ALDX', 'MVST'] # enter whichever tickers you want to look at...
        else:
            print('USER INPUT ERROR: Please type "DONE" and press "ENTER" key when finished inputting stocks of interest')
    # API call by ticker to get insider trading by symbol https://financialmodelingprep.com/api/v4/insider-trading?symbol= + str(tick_list) + '&page=0'
    

# Execute sequence of function calls to run program
setup()

# Read API Keys from Config File
with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
TOKEN = cfg["TOKEN"]

find_sib_stocks()
gen_sib_positive_pv_graphs()
