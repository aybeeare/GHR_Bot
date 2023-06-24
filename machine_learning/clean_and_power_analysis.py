# Cleanup and Power Analysis of Stock Dataset

import pandas as pd
from statsmodels.stats.power import TTestIndPower


def dataset_cleanup():
    df = pd.read_csv('ML_Dataset.csv')

    # Get rid of unwanted columns
    unwanted_col = ['Unnamed: 0']
    # unwanted_col = ['Insider', 'Representative', 'Senator', 
    #                 'operatingCashFlowPerShare.1','freeCashFlowPerShare.1','cashPerShare.1', 'priceToSalesRatio.1', 
    #                 'currentRatio.1','interestCoverage.1', 'dividendYield.1','payoutRatio.1', 
    #                 'receivablesTurnover.1', 'payablesTurnover.1', 'inventoryTurnover.1']

    df = df.drop(labels = unwanted_col, axis = 1)

    days_payable_values = df['daysOfPayablesOutstanding'].tolist() # has a difference between 0.0 and nan so if this is nan, get rid of it because pull failed

    # Remove data point by index if feat "daysOfPayablesOutstanding" is nan
    idx_list = []
    for idx in range(len(days_payable_values)):

        if pd.isna(days_payable_values[idx]):
            idx_list.append(idx)

    df = df.drop(labels = idx_list, axis = 0)
    df = df.fillna(0) # fill remaining nan with 0
    df = df.sample(frac = 1)

    print(df)
    df.to_csv('ML_Dataset_Clean.csv')

def power_analysis():
    power_analysis = TTestIndPower()

    # Parameters for analysis (typical accepted values for experiment)

    effect = 0.8 
    alpha = 0.05
    power = 0.9

    nobs = power_analysis.solve_power(effect, power = power, nobs1 = None, alpha = alpha, ratio = 1.0)
    print('Number of Observations Required to Achieve Standard: ', nobs)

# Main script:

# dataset_cleanup()
# power_analysis()

df = pd.read_csv('ML_Dataset_Clean.csv')
print(df)