# GHR Bot gathers significant insider buys for a specified year, and collects common financial metrics 
# for the stock at the time of the buy. This script is used for creating a dataset of signficant insider
# buys over the years, with the intention of applying machine learning algorithms to this dataset to model
# which insider buy stocks to select for best investment strategy :)

from preprocessing import new_buys_format
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
import time

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
    print('Done List All Stocks!')

# Find all sibs for all stocks within the past year months
def recent_sibs(test_year):

    # Find current date and time to check how recent and only run this code periodically and allow for keyboard interrupts
    date = str(datetime.date.today()).split('-')
    global yr, mth, day
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
        count_high = False

        #print('Tick: ', tick)
        time.sleep(0.25) # too many requests per minute error, slow it down.
        API_Key_Insider = API_KEY_INSIDER_BUYING + str(tick) + '&page=0&apikey=' + TOKEN
        #print('API KEY: ', API_Key_Insider)

        # Pull Insider Trading Data from API
        req = urllib.request.Request(API_Key_Insider) # instantiate request
        response = urllib.request.urlopen(req) # send request to API
        res_body = response.read().decode('utf-8') # read response
        filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)
        
        for dics in filingsJson:
            #print(dics)
            try:
                tick = dics.get('symbol')
                trans_date = dics.get('transactionDate')
                year, month, day = trans_date.split('-')
                
                trans_type = dics.get('transactionType')[0] # 'S-Sale' or 'P-Purchase'
                trans_amount =  (float(dics.get('price')))*(float(dics.get('securitiesTransacted')))
                insider = dics.get('reportingName')
                security = dics.get('securityName')
            
            except:

                #print('Continued!', count)
                if count > 12000:
                    #("Went in")
                    count_high = True
                    #df.to_csv('ghr_bot_sib_record2.csv')
                    break # break inner loop

            if trans_type == 'P' and trans_amount > 250000 and 'Common' in security and (int(yr) - int(year) == test_year): #and ((int(month) - int(mth)) >= 6)):
                if tick not in symbol_list or trans_date not in trans_list or insider not in reg_insider_list:
                    df.loc[len(df)] = [tick, insider, None, trans_date, trans_amount, None, None, None, None, None, None, None, None, None, None] # append new row to end of df
                    symbol_list.append(tick)
                    trans_list.append(trans_date)
                    reg_insider_list.append(insider)
        
        if count_high: # once inner loop done or broken out of, check if count_high was asserted, if it was, break and finish
            break 
        
    # Write df to CSV
    #print(df)
    df.to_csv('ghr_bot_sib_record2.csv')
    print('Done Recent Sibs!')

# Find stocks with significant insider buying (transactions > $300,000)
def current_sib_stocks(months):

    # Find current date and time to check how recent and only run this code periodically and allow for keyboard interrupts
    date = str(datetime.date.today()).split('-')
    global yr, mth, day
    yr, mth, day = date
    
    # Instantiate empty dataframe containing symbol, trans_type, trans_amount, trans_date, buyer
    COLUMN_NAMES = ['Symbol','Insider', 'Ins-Count', 'Ins-Date', 'Ins-Amt', 'Senator', 'Sen-Date', 'Sen-Amt','Representative', 'Rep-Date', 'Rep-Amt', 'Press', 'Bargain', 'PV Trend', 'Buy Strength']
    
    df = pd.DataFrame(columns=COLUMN_NAMES)
    symbol_list = []
    trans_list = []
    reg_insider_list = []

    API_KEY_INSIDER_BUYING = cfg["API_KEY_INSIDER_TICK"]
    

    for tick in stocks_tup:

        # Pull Insider Trading Data from API
        time.sleep(0.05) # too many requests per minute error, slow it down.
        API_Key_Insider = API_KEY_INSIDER_BUYING + str(tick) + '&page=0&apikey=' + TOKEN
        req = urllib.request.Request(API_Key_Insider) # instantiate request
        response = urllib.request.urlopen(req) # send request to API
        res_body = response.read().decode('utf-8') # read response
        filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)

        for dics in filingsJson:
            
            try:
                tick = dics.get('symbol')
                trans_type = dics.get('transactionType')[0] # 'S-Sale' or 'P-Purchase'
                trans_amount =  (float(dics.get('price')))*(float(dics.get('securitiesTransacted')))
                trans_date = dics.get('transactionDate')
                year, month, day = trans_date.split('-')
                insider = dics.get('reportingName')
                security = dics.get('securityName')

            except:
                continue
            
            if trans_type == 'P' and trans_amount > 300000 and 'Common' in security and (((int(mth) - int(month) <= months) and yr == year) or (int(yr) - int(year) == 1 and ((int(month) - int(mth)) >= (12 - months)))):
                if tick not in symbol_list or trans_date not in trans_list or insider not in reg_insider_list:
                    df.loc[len(df)] = [tick, insider, None, trans_date, trans_amount, None, None, None, None, None, None, None, None, None, None]# append new row to end of df
                    symbol_list.append(tick)
                    trans_list.append(trans_date)
                    reg_insider_list.append(insider)
    
    df_str = 'recents_' + str(yr) + '_' + str(mth) + '_' + str(day)
    df.to_csv(df_str)
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
            try:
                date = int(cleanup_dict[sym][2][idx].replace('-','')) # broke with strange date
            except:
                continue

            if date > most_rec:
                most_rec = date
                most_rec_idx = idx

        cleanup_dict[sym][0] = cleanup_dict[sym][0][most_rec_idx] # select corresponding insider w/ most recent date
        cleanup_dict[sym][2] = cleanup_dict[sym][2][most_rec_idx]
        cleanup_dict[sym][3] = int(cleanup_dict[sym][3] / cleanup_dict[sym][1]) # avg trans = cum/count

    # Instantiate empty dataframe containing symbol, trans_type, trans_amount, trans_date, buyer

    COLUMN_NAMES = ['Insider', 'Ins-Count', 'Ins-Date', 'Ins-Amt'] # ... 'Representative', 'Rep-Date', 'Rep-Amt', 'Senator', 'Sen-Date', 'Sen-Amt', 'Press', 'Bargain', 'PV Trend', 'Buy Strength']

    df_clean = pd.DataFrame.from_dict(cleanup_dict, orient='index', columns=COLUMN_NAMES)
    df_clean.to_csv('ghr_bot_sib_record2_cp.csv')    
    print('Done DF Cleanup!') 

# Extend insiders to track members of senate and house and consolidate in dataframe
def sib_extend_politicians():

    # Read ghr_bot_sib_record2_cp.csv as df and append new sib stocks from politicians
    df = pd.read_csv('ghr_bot_sib_record2_cp.csv')
    df.rename( columns={'Unnamed: 0' :'Symbol'}, inplace=True)
    symbol_list = df['Symbol'].tolist()

    # Instantiate Empty Lists for House and Senate
    rep, rep_date, rep_amt, sen, sen_date, sen_amt = ([] for i in range(6))
    
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

        if (((int(mth) - int(month) <= 3) and yr == year) or (int(yr) - int(year) == 1 and ((int(month) - int(mth)) >= 9))) and ('purchase' in x['type']): # logic for less than one year
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
    
    for tick in symbol_list: # Still only going for stocks that insiders bought in last 3 months, politicians alone not enough
        if tick in house_dict.keys() or tick in senate_dict.keys():
            if tick in house_dict.keys() and tick not in senate_dict.keys():

                # Find most recent date from list of dates
                most_rec = 0
                most_rec_idx = 0

                for idx in range(len(house_dict[tick][1])):
                    date = int(house_dict[tick][1][idx].translate({ord(c): "" for c in ",-/"}))

                    if date > most_rec:
                        most_rec = date
                        most_rec_idx = idx

                current_larg = 0

                # Find largest amount in list of amounts (strings)
                
                for amt in house_dict[tick][2]:
                    amt = amt.translate({ord(c): "" for c in "$,-"})
                    amt = int(amt.split(" ")[0])
                    
                    if amt > current_larg:
                        current_larg = amt

                #print(tick, amt)
                rep.append([house_dict[tick][0][most_rec_idx],len(house_dict[tick][0])])
                rep_date.append(house_dict[tick][1][most_rec_idx])
                rep_amt.append(current_larg) 
                sen.append(0)
                sen_date.append(0)
                sen_amt.append(0)
            
            elif tick in senate_dict.keys() and tick not in house_dict.keys():

                # Find most recent date from list of dates
                most_rec = 0
                most_rec_idx = 0

                for idx in range(len(senate_dict[tick][1])):
                    date = int(senate_dict[tick][1][idx].translate({ord(c): "" for c in ",-/"}))

                    if date > most_rec:
                        most_rec = date
                        most_rec_idx = idx

                current_larg = 0

                # Find largest amount in list of amounts (strings)
                
                for amt in senate_dict[tick][2]:
                    amt = amt.translate({ord(c): "" for c in "$,-"})
                    amt = int(amt.split(" ")[0])
                    
                    if amt > current_larg:
                        current_larg = amt

                #print(tick, amt)
                sen.append([senate_dict[tick][0][most_rec_idx],len(senate_dict[tick][0])])
                sen_date.append(senate_dict[tick][1][most_rec_idx])
                sen_amt.append(current_larg)
                rep.append(0)
                rep_date.append(0)
                rep_amt.append(0)

            else: # tick in both

                # Find most recent date from list of dates
                most_rec = 0
                most_rec_idx = 0

                for idx in range(len(house_dict[tick][1])):
                    date = int(house_dict[tick][1][idx].translate({ord(c): "" for c in ",-/"}))

                    if date > most_rec:
                        most_rec = date
                        most_rec_idx = idx

                current_larg = 0

                # Find largest amount in list of amounts (strings)
                
                for amt in house_dict[tick][2]:
                    amt = amt.translate({ord(c): "" for c in "$,-"})
                    amt = int(amt.split(" ")[0])
                    
                    if amt > current_larg:
                        current_larg = amt   

                rep.append([house_dict[tick][0][most_rec_idx],len(house_dict[tick][0])])
                rep_date.append(house_dict[tick][1][most_rec_idx])
                rep_amt.append(current_larg)

                # Find most recent date from list of dates
                most_rec = 0
                most_rec_idx = 0

                for idx in range(len(senate_dict[tick][1])):
                    date = int(senate_dict[tick][1][idx].translate({ord(c): "" for c in ",-/"}))

                    if date > most_rec:
                        most_rec = date
                        most_rec_idx = idx

                current_larg = 0

                # Find largest amount in list of amounts (strings)
                
                for amt in senate_dict[tick][2]:
                    amt = amt.translate({ord(c): "" for c in "$,-"})
                    amt = int(amt.split(" ")[0])
                    
                    if amt > current_larg:
                        current_larg = amt

                sen.append([senate_dict[tick][0][most_rec_idx],len(senate_dict[tick][0])])
                sen_date.append(senate_dict[tick][1][most_rec_idx])
                sen_amt.append(current_larg)
                #print(tick, amt)

        else: # tick not in either house or senate
            rep.append(0)
            rep_date.append(0)
            rep_amt.append(0)
            sen.append(0)
            sen_date.append(0)
            sen_amt.append(0)

    # Update DF
    df['Representative'] = rep
    df['Rep-Date'] = rep_date
    df['Rep-Amt'] = rep_amt
    df['Senator'] = sen
    df['Sen-Date'] = sen_date
    df['Sen-Amt'] = sen_amt

    df.to_csv('ghr_bot_sib_record_clean.csv')  
    print('Done Politician Extension!')                

        
def sib_extend_fundamental_ratios():

    df1 = pd.read_csv('ghr_bot_sib_record_clean.csv')
    #print(df1)
    symbol_list = df1['Symbol'].tolist()
    date_list = df1['Ins-Date'].tolist()

    df2_columns = ['symbol', 'date', 'period', 'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding', 'daysOfInventoryOutstanding', 'operatingCycle', 'daysOfPayablesOutstanding', 'cashConversionCycle', 'grossProfitMargin', 'operatingProfitMargin',                                                                                                     
                   'pretaxProfitMargin', 'netProfitMargin', 'effectiveTaxRate', 'returnOnAssets', 'returnOnEquity', 'returnOnCapitalEmployed', 'netIncomePerEBT', 'ebtPerEbit', 'ebitPerRevenue', 'debtRatio', 'debtEquityRatio', 'longTermDebtToCapitalization', 
                   'totalDebtToCapitalization', 'interestCoverage', 'cashFlowToDebtRatio', 'companyEquityMultiplier', 'receivablesTurnover', 'payablesTurnover', 'inventoryTurnover', 'fixedAssetTurnover', 'assetTurnover', 'operatingCashFlowPerShare', 'freeCashFlowPerShare',  
                   'cashPerShare', 'payoutRatio', 'operatingCashFlowSalesRatio', 'freeCashFlowOperatingCashFlowRatio', 'cashFlowCoverageRatios', 'shortTermCoverageRatios', 'capitalExpenditureCoverageRatio', 'dividendPaidAndCapexCoverageRatio', 'dividendPayoutRatio', 'priceBookValueRatio', 
                   'priceToBookRatio', 'priceToSalesRatio', 'priceEarningsRatio', 'priceToFreeCashFlowsRatio', 'priceToOperatingCashFlowsRatio', 'priceCashFlowRatio', 'priceEarningsToGrowthRatio', 'priceSalesRatio', 'dividendYield', 'enterpriseValueMultiple', 'priceFairValue']
    
    df2 = pd.DataFrame(columns = df2_columns)
    count = 0

    # Create new data frame and at the end, append to old dataframe
    

    for tick, ins_date, i in zip(symbol_list, date_list, range(len(symbol_list))):

        try:
            print('Tick: ', tick)
            # print('Insider Date: ', ins_date)
            # print('############################')
            API_KEY_RATIOS = cfg["API_KEY_RATIOS"]
            API_Key_Insider = API_KEY_RATIOS + str(tick) + '?period=quarter&limit=140&apikey=' + TOKEN 

            # Pull Financial Ratios for Tick
            req = urllib.request.Request(API_Key_Insider) # instantiate request
            response = urllib.request.urlopen(req) # send request to API
            res_body = response.read().decode('utf-8') # read response
            filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)
            #print(filingsJson)
            #count += 1

        except:
            print('Tick Continued: ', tick)
            print('Continued')
            continue
        
        for dics in filingsJson:
            
            ins_yr, ins_mon, ins_day = [int(x) for x in ins_date.split('-')]
            qtr_yr, qtr_mon, qtr_day = [int(x) for x in dics.get('date').split('-')]
            
            
            # Logic for quarter month in last 3 months or its the new year and last 2 quarters (not ideal want most recent but kinda limited by tools here) was december of previous year
            if (ins_yr == qtr_yr and (ins_mon >= qtr_mon and (ins_mon - qtr_mon) <= 3)) or ((ins_yr == qtr_yr + 1) and (qtr_mon - 9 >= ins_mon)):

                # Create df2 on first iteration
                #print(dics.keys())
                
                    
                
                # print('Ticker: ', tick)
                # print('Insider Date', ins_date)
                # print('Quarter Date', dics.get('date'))
            
                #print(type(dics))

                df2.loc[i] = dics
                

            else:
                continue
    
    #print('Count', count)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('ML_Dataset_Feats1.csv') 
    print('Done Fundamental Ratios!') 


def sib_extend_fundamental_metrics():

    df1 = pd.read_csv('ML_Dataset_Feats1.csv')
    #print(df1)
    symbol_list = df1['Symbol'].tolist()
    date_list = df1['Ins-Date'].tolist() 
    df2_columns = ['symbol', 'date', 'period', 'revenuePerShare', 'netIncomePerShare', 'operatingCashFlowPerShare', 'freeCashFlowPerShare', 
                   'cashPerShare', 'bookValuePerShare', 'tangibleBookValuePerShare', 'shareholdersEquityPerShare', 'interestDebtPerShare', 'marketCap',
                   'enterpriseValue', 'peRatio', 'priceToSalesRatio', 'pocfratio', 'pfcfRatio', 'pbRatio', 'ptbRatio', 'evToSales', 'enterpriseValueOverEBITDA',
                   'evToOperatingCashFlow', 'evToFreeCashFlow', 'earningsYield', 'freeCashFlowYield', 'debtToEquity', 'debtToAssets', 'netDebtToEBITDA', 'currentRatio', 
                   'interestCoverage', 'incomeQuality', 'dividendYield', 'payoutRatio', 'salesGeneralAndAdministrativeToRevenue', 'researchAndDdevelopementToRevenue', 'intangiblesToTotalAssets', 
                   'capexToOperatingCashFlow', 'capexToRevenue', 'capexToDepreciation', 'stockBasedCompensationToRevenue', 'grahamNumber', 'roic', 'returnOnTangibleAssets', 'grahamNetNet', 'workingCapital', 
                   'tangibleAssetValue', 'netCurrentAssetValue', 'investedCapital', 'averageReceivables', 'averagePayables', 'averageInventory', 'daysSalesOutstanding', 'daysPayablesOutstanding', 'daysOfInventoryOnHand', 
                   'receivablesTurnover', 'payablesTurnover', 'inventoryTurnover', 'roe', 'capexPerShare']
    
    df2 = pd.DataFrame(columns = df2_columns)
    #print(symbol_list)   
    count = 0

    # Create new data frame and at the end, append to old dataframe
    

    for tick, ins_date, i in zip(symbol_list, date_list, range(len(symbol_list))):

        try:
            print('Tick: ', tick)
            API_KEY_METRICS = cfg["API_KEY_METRICS"]
            API_Key_Insider = API_KEY_METRICS + str(tick) + '?period=quarter&limit=130&apikey=' + TOKEN 

            # Pull Financial Ratios for Tick
            req = urllib.request.Request(API_Key_Insider) # instantiate request
            response = urllib.request.urlopen(req) # send request to API
            res_body = response.read().decode('utf-8') # read response
            filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)
            #print(filingsJson)
            

        except:
            #print('Continued')
            print('Tick Continued: ', tick)
            continue
        
        for dics in filingsJson:
            
            ins_yr, ins_mon, ins_day = [int(x) for x in ins_date.split('-')]
            qtr_yr, qtr_mon, qtr_day = [int(x) for x in dics.get('date').split('-')]
            
            
            # Logic for quarter month in last 3 months or its the new year and last 2 quarters (not ideal want most recent but kinda limited by tools here) was december of previous year
            if (ins_yr == qtr_yr and (ins_mon >= qtr_mon and (ins_mon - qtr_mon) <= 3)) or ((ins_yr == qtr_yr + 1) and (qtr_mon - 9 >= ins_mon)):
                
                

                
                # print('Ticker: ', tick)
                # print('Insider Date', ins_date)
                # print('Quarter Date', dics.get('date'))
                #print(type(dics))

                df2.loc[i] = dics
                

            else:
                continue
    
    #print('Count', count)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('ML_Unlabeled_Current.csv')  
    print('Done Fundamental Metrics!')

# Find 3, 6, and 12 month returns for each stock in list, and label
def read_and_label(test_year):

    df1 = pd.read_csv('ML_Unlabeled.csv')
    df1 = df1.drop(df1.columns[[0,1,2,14,15,16,71,72,73]], axis = 1)
    #df1.to_csv('test.csv')  
    symbol_list = df1['Symbol'].tolist()
    date_list = df1['Ins-Date'].tolist() 

    mon0_dic= {} # price at time of insider buy
    mon3_dic = {}
    mon6_dic = {}
    mon12_dic = {}
    count = 0

    for tick, ins_date, i in zip(symbol_list, date_list, range(len(symbol_list))):

        # Get 3, 6, and 12 month returns for tick
        try:
            #print('Tick: ', tick)
            #print('Insider Date: ', ins_date)
            ins_yr, ins_mon, ins_day = ins_date.split('-')
            ins_yr = int(ins_yr)
            ins_yr_next = ins_yr + 1
            API_DAILY = cfg["API_DAILY"]
            API_Key_Insider = API_DAILY + str(tick) + '?from=' + str(ins_date) + '&to=' + str(ins_yr_next) + '-' + ins_mon + '-' + ins_day + '&apikey=' + TOKEN 

            # Pull Daily Price History for Tick
            req = urllib.request.Request(API_Key_Insider) # instantiate request
            response = urllib.request.urlopen(req) # send request to API
            res_body = response.read().decode('utf-8') # read response
            filingsJson = json.loads(res_body) # transform response into json (data is list of dictionaries)

            # Get 3, 6, and 12 months after insider buy in string form to pull from list of dictionaries
            ins_mon_str = ins_mon

            ins_mon = int(ins_mon)
            ins_mon3 = ins_mon + 3
            ins_mon6 = ins_mon + 6

            # Convert to 2 digit string 
            if ins_mon3 < 10:
                ins_mon3 = '0' + str(ins_mon3)
                ins_yr3 = str(ins_yr)
            
            elif ins_mon3 > 12: 
                ins_mon3 = '0' + str(ins_mon3 - 12)
                ins_yr3 = str(ins_yr + 1)
            
            else:
                ins_mon3 = str(ins_mon3)
                ins_yr3 = str(ins_yr)

            
            if ins_mon6 < 10:
                ins_mon6 = '0' + str(ins_mon6)
                ins_yr6 = str(ins_yr)
            
            elif ins_mon6 > 12: 
                ins_mon6 = '0' + str(ins_mon6 - 12)
                ins_yr6 = str(ins_yr + 1)
            
            else:
                ins_mon6 = str(ins_mon6)
                ins_yr6 = str(ins_yr)
            
            ins_date3 = ins_yr3 + '-' + ins_mon3 + '-' + str(ins_day)
            ins_date6 = ins_yr6 + '-' + ins_mon6 + '-' + str(ins_day)
            ins_date12 = str(ins_yr_next) + '-' + ins_mon_str + '-' + str(ins_day)
            
            # Make sure dictionaries are same size even if it pulls different numbers of mon3, mon6, mon12 data
            mon3 = False
            mon6 = False
            mon12 = False

            mon0_dic[tick] = filingsJson['historical'][len(filingsJson['historical']) -1]['close']
            for i in range(len(filingsJson['historical'])):

                if ins_date3 == filingsJson['historical'][i]['date']: 
                    mon3_dic[tick] = filingsJson['historical'][i]['close']
                    mon3 = True
                    # print('List Index: ', i)
                    # print('Ins_date3: ', ins_date3)
                    # print(filingsJson['historical'][i])
                    
                
                if ins_date6 == filingsJson['historical'][i]['date']: 
                    mon6_dic[tick] = filingsJson['historical'][i]['close']
                    mon6 = True
                    # print('List Index: ', i)
                    # print('Ins_date6: ', ins_date6)
                    # print(filingsJson['historical'][i])
                    
                
                if ins_date12 == filingsJson['historical'][i]['date']: 
                    mon12_dic[tick] = filingsJson['historical'][i]['close']
                    mon12 = True
                    # print('List Index: ', i)
                    # print('Ins_date12: ', ins_date12)
                    # print(filingsJson['historical'][i])
                
            if not mon3:
                mon3_dic[tick] = ''
            
            if not mon6:
                mon6_dic[tick] = ''
            
            if not mon12:
                mon12_dic[tick] = ''
            
            
        except:
            #print('Continued')
            continue
    
    # Find indices of differences between two lists
    drop_syms = []
    drop_idx = []

    for sym, idx in zip(symbol_list, range(len(symbol_list))):

        if sym not in mon0_dic.keys():
            drop_idx.append(idx)
    
    # print('Dropped Symbols: ', drop_syms)
    # print('Dropped Indices: ', drop_idx)
    df1 = df1.drop(labels = drop_idx, axis = 0)
    drop_idx.clear()
    #print(df1)
    
    # Classify as weak buy (- return), buy (+ return > 5%), strong buy (+ return > 10%) on dataframe with removed indices
    symbol_list = df1['Symbol'].tolist()
    buy_rec = {}

    for sym in symbol_list: # iterate through new df ticks
        ins_buy_price = mon0_dic[sym] 
        # attempt to get return for 3, 6, and 12 month
        mon3_price = mon3_dic[sym]
        if mon3_price != '':
            mon3_return = (mon3_price - ins_buy_price)/ins_buy_price
        else:
            mon3_return = 0

        mon6_price = mon6_dic[sym]
        if mon6_price != '':
            mon6_return = (mon6_price - ins_buy_price)/ins_buy_price
        else:
            mon6_return = 0
        
        mon12_price = mon12_dic[sym]
        if mon12_price != '':
            mon12_return = (mon12_price - ins_buy_price)/ins_buy_price
        else:
            mon12_return = 0
        
        if (mon3_return > 0.10) or (mon6_return > 0.10) or (mon12_return > 0.10):
            buy_rec[sym] = 'Strong Buy'
        
        elif (mon3_return > 0.05) or (mon6_return > 0.05) or (mon12_return > 0.05):
            buy_rec[sym] = 'Buy'
        
        else:
            buy_rec[sym] = 'Weak'
    
    # Quality Check :)
    # print('Dataframe Symbol List: ', len(symbol_list))
    # print('Mon3 List: ', len(mon3_dic.keys()))
    # print('Mon6 List: ', len(mon6_dic.keys()))
    # print('Mon12 List: ', len(mon12_dic.keys()))
    # print('Buy Rec List: ', len(buy_rec.keys()))
    # print('Buy Recs: ', len(buy_rec))

    mon0_values = list(mon0_dic.values())
    df1['Ins-Buy-Price'] = mon0_values
    mon3_values = list(mon3_dic.values())
    mon3_values = [0 if x == '' else x for x in mon3_values] # replace all '' strings with '0'
    df1['Mon-3-Price'] = mon3_values
    mon6_values = list(mon6_dic.values())
    mon6_values = [0 if x == '' else x for x in mon6_values] # replace all '' strings with '0'
    df1['Mon-6-Price'] = mon6_values
    mon12_values = list(mon12_dic.values())
    mon12_values = [0 if x == '' else x for x in mon12_values] # replace all '' strings with '0'
    df1['Mon-12-Price'] = mon12_values
    df1['Labels'] = list(buy_rec.values())
    

    # Remove indices using empty drop_idx list
    for i in range(len(mon0_values)):
        if mon3_values[i] == 0 and mon6_values[i] == 0 and mon12_values[i] == 0:
            drop_idx.append(i)
    
    for idx in drop_idx:
        try:
            df1 = df1.drop(labels = [idx], axis = 0) # account for trying to remove already removed row
        except:
            continue

    csv_str = 'Year_Data' + str(test_year) + '.csv'
    df1.to_csv(csv_str)
    print('Done Read and Label!')
    

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

        # try:
        #     plt.title('PV for Significant Insider Buying Stock: '+ str(tick_list[index])) 
        #     plt.plot(cum_vol_points, price_points)
        #     plt.xlabel('Volume (millions of shares)')
        #     plt.ylabel('Price ($/share)')
        #     plt.show()
        # except:
        #     print('Could not plot: ', tick_list[index])
        #     continue
    
        # if ticker_index == range(len(tick_list) -1)[-1]:
        #     print('\n\n\n\n\n\n\n\n\nDONE!\nTHANK YOU FOR CHOOSING THE GHR BOT!\nYOURS,\nABR')

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
    pass

def gen_portfolio_management_df():
    pass

def recommendation_algo(): 
    pass

# Execute main script to generate csv files for each of the following years ago, specified in recent_sibs fcn

def build_dataset():

    years_ago_list = [1] # Takes list of "years_ago" to check, e.g. 9 would be 2023 - 9 = 2014, gets all insiders this year.

    for years_ago in years_ago_list:

        start_time = time.time()
        list_all_stocks()
        try:
            recent_sibs(years_ago) # current_sib_stocks()
        except:
            continue           
        df_cleanup()
        sib_extend_politicians()
        sib_extend_fundamental_ratios()
        sib_extend_fundamental_metrics()
        read_and_label(years_ago)

        end_time = time.time()
        print('Years Ago Done!: ', years_ago)
        print('Execution Time (s)', end_time - start_time)

# Function that fetches current sibs within specified number of months ago 
def fetch_current(months):

    # list_all_stocks()
    # #recent_sibs(test_year)
    # current_sib_stocks(months)
    # df_cleanup()
    # sib_extend_politicians()
    # sib_extend_fundamental_ratios()
    sib_extend_fundamental_metrics()
    new_buys_format()


# def main():
#     fetch_current(0) # fetch recents for current year (0 years back)

# if __name__ == '__main__':
#     main()

#generate_pv_and_plot(test_list)
#generate_pv_and_classify(test_list)


