import pandas as pd
import numpy as np  

# Preprocessing methods including initial cleanup of csv, normalization methods, and analyzing dataset variance

def train_cleanup():
    # Preliminary feature screen and remove labels (ticker name, date, and labels) so remaining dataset can be normalized
    df = pd.read_csv('ML_Dataset_Clean.csv')
    
    # Keep track of these features to be added to dataset at the end
    symbol_list = df['Symbol'].tolist()
    date_list = df['Ins-Date'].tolist()
    labels = df['Labels'].tolist()
    
    # Update Labels so Weak = 0, Buy = 1, and Strong Buy = 2
    labels = [0 if element == 'Weak' else element for element in labels]
    labels = [1 if element == 'Buy' else element for element in labels]
    labels = [2 if element == 'Strong Buy' else element for element in labels]
    df['Labels'] = labels
    
    unwanted_col = ['Rep-Date', 'Rep-Amt', 'Symbol', 'Ins-Date',
                    'Sen-Date', 'Sen-Amt', 'Ins-Buy-Price', 'Mon-1-Price', 
                    'Mon-2-Price', 'Mon-3-Price', 'Mon-6-Price' , 
                    'Mon-9-Price', 'Mon-12-Price']
    
    df = df.drop(labels = unwanted_col, axis = 1)
    # print('Training Set')
    # print(df)
    unnormalized_data = df.to_numpy()
    return unnormalized_data

# After collecting new insider buys to be fed into ML framework (stored in ML_Unlabeled.csv), clean to fit model.
def new_buys_format():

    df = pd.read_csv('ML_Unlabeled_Current.csv')
    
    unwanted_col = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2', 'symbol', 'date', 'period', 'symbol.1', 'date.1', 'period.1', 'Insider', 
                    'Representative', 'Senator', 'operatingCashFlowPerShare.1','freeCashFlowPerShare.1','cashPerShare.1', 'priceToSalesRatio.1', 
                'currentRatio.1','interestCoverage.1', 'dividendYield.1','payoutRatio.1', 'receivablesTurnover.1', 'payablesTurnover.1', 
                'inventoryTurnover.1', 'Rep-Date', 'Rep-Amt', 'Ins-Date',
                'Sen-Date', 'Sen-Amt']

    df = df.drop(labels = unwanted_col, axis = 1) # Get rid of unwanted columns  
    
    days_payable_values = df['daysOfPayablesOutstanding'].tolist() # has a difference between 0.0 and nan so if this is nan, get rid of it because pull failed

    # Remove data point by index if feat "daysOfPayablesOutstanding" is nan
    idx_list = []
    for idx in range(len(days_payable_values)):

        if pd.isna(days_payable_values[idx]):
            idx_list.append(idx)

    df = df.drop(labels = idx_list, axis = 0)
    df = df.fillna(0) # fill remaining nan with 0
    new_buys_df = df.sample(frac = 1)
    
    test_symbols = new_buys_df['Symbol'].tolist()
    new_buys_df = new_buys_df.drop(labels = 'Symbol', axis = 1)
    df.to_csv('Unlabeled_Test_Fixed.csv')
    # print('Full Dataset\n')
    # print(new_buys_df)
    new_buys = new_buys_df.to_numpy()

    return new_buys, test_symbols

# Normalization
def min_max_normalization(unnormalized_data):
    #rank = np.linalg.matrix_rank(unnormalized_data)
    normalized = np.zeros((unnormalized_data.shape[0], unnormalized_data.shape[1])) # initialize empty matrix to fill with normalized values
    
    # Get min and max of each feature and normalize following max/min method (normalize all features except labels)
    for f in range(unnormalized_data.shape[1] - 1):
        
        max_f = np.max(unnormalized_data[:, f])
        min_f = np.min(unnormalized_data[:, f])
        
        
        max_vector = max_f*np.ones((unnormalized_data.shape[0],))
        min_vector = min_f*np.ones((unnormalized_data.shape[0],))
        diff_vector = max_vector - min_vector
        
        # Perform element wise (feature data - min)/(max - min) to do max/min normalization
        num = unnormalized_data[:, f] - min_vector
        normalized[:, f] = np.divide(num, diff_vector)
    
    normalized[:, -1] = unnormalized_data[:, -1] # retain same labels in normalized 
    return normalized
    
        
def z_normalization(unnormalized_data):
    
    normalized = np.zeros((unnormalized_data.shape[0], unnormalized_data.shape[1])) # initialize empty matrix to fill with normalized values
    
    # Get min and max of each feature and normalize following max/min method (normalize all features except labels)
    for f in range(unnormalized_data.shape[1] - 1):
        
        mean_f = np.mean(unnormalized_data[:, f])
        std_f = np.std(unnormalized_data[:, f])
        
        mean_vector = mean_f*np.ones((unnormalized_data.shape[0],))
        std_vector = std_f*np.ones((unnormalized_data.shape[0],))
        
        # Perform element wise (feature data - mean)/std to do z score normalization (best for handling outliers)
        num = unnormalized_data[:, f] - mean_vector
        normalized[:, f] = np.divide(num, std_vector)
    
    normalized[:, -1] = unnormalized_data[:, -1] # retain same labels in normalized 
    return normalized

def get_dataset_stats(unnormalized_data):
    column_var = np.var(unnormalized_data, axis = 0)
    col_0_var = np.var(unnormalized_data[:, 0])
    print('Col 0 Var: ', col_0_var)
    print('Col Var: ', column_var)
    
    return column_var

new_buys_format()