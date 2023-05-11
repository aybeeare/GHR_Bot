import pandas as pd
import numpy as np  

# Preprocessing methods including initial cleanup of csv, normalization methods, and analyzing dataset variance

def cleanup():
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
    unnormalized_data = df.to_numpy()
    return unnormalized_data

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

