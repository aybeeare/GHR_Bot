import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn import svm
from mrmr import mrmr_classif
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define classifier class with classifier methods to test and filters to apply to input data
    
    
class Classifier(nn.Module): # Has methods calling particular classifier and all filtering methods
    
    def __init__(self, normalized_data):
        
        self.normalized_data = normalized_data
        
        
    def xg_boost(self, X_train, y_train, X_test, y_test): 
    
        xgbc = XGBClassifier(objective = 'multi:softmax', num_class = 3)   
        xgbc.fit(X_train, y_train)
        
        # Get Cross Validation Score
        # scores = cross_val_score(xgbc, X_train, y_train, cv=5)
        # print("Mean cross-validation score: %.2f" % scores.mean())
        
        # Get K-Validation Score
        # kfold = KFold(n_splits=10, shuffle=True)
        # kf_cv_scores = cross_val_score(xgbc, X_train, y_train, cv=kfold )
        # print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
        
        y_pred_test = xgbc.predict(X_test)
        y_pred_train = xgbc.predict(X_train)
        # accuracy = accuracy_score(y_pred_test, y_test) # future idea to collect good models and save
        
        # if accuracy >= 0.55:
            
        #     xgbc.save_model("model.json")
        # print("Accuracy score: %.2f" % accuracy)
        
        return y_pred_test, y_pred_train
    
    def cat_boost(self, X_train, y_train, X_test, y_test): 
        
        cat_model = CatBoostClassifier(
                               loss_function='MultiClass',
                               verbose=True)   
        
        cat_model.fit(X_train, y_train)
        
        # Get Cross Validation Score
        # scores = cross_val_score(cat_model, X_train, y_train, cv=5)
        # print("Mean cross-validation score: %.2f" % scores.mean())
        
        # Get K-Validation Score
        # kfold = KFold(n_splits=10, shuffle=True)
        # kf_cv_scores = cross_val_score(cat_model, X_train, y_train, cv=kfold )
        # print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
        
        y_pred_test = cat_model.predict(X_test)
        y_pred_train = cat_model.predict(X_train)
        # accuracy = accuracy_score(y_pred, y_test)
        # print("Accuracy score: %.2f" % accuracy)
        # print("Mean cross-validation score: %.2f" % scores.mean())
        
        return y_pred_test, y_pred_train
    
        
    def knn(self, X_train, y_train, X_test, y_test, k_neighbors): # (self, feats_selected, feat_select_filter, pca, X_reduced, k_neighbors): 
    
        # originally did split inside function which was pretty nepic, just    
    
        # # split data into training and test sets 
        # normalized = self.normalized_data
        # y = normalized[:, -1] # select last column
        # X = normalized[:, :-1] # select all columns removing last
        
        # if pca == True:
        #     X = X_reduced
        
        # if feat_select_filter == True:
        #     X = normalized[:, feats_selected]
       
        knn = KNeighborsClassifier(n_neighbors= k_neighbors)
        knn.fit(X_train, y_train)
        
        # Get Cross Validation Score
        # scores = cross_val_score(knn, X_train, y_train, cv=5)
        # print("Mean cross-validation score: %.2f" % scores.mean())
        
        # Get K-Validation Score
        # kfold = KFold(n_splits=10, shuffle=True)
        # kf_cv_scores = cross_val_score(knn, X_train, y_train, cv=kfold )
        # print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
        
        y_pred_test = knn.predict(X_test)
        y_pred_train = knn.predict(X_train)
        # accuracy = accuracy_score(y_pred, y_test)
        # print("Accuracy score: %.2f" % accuracy)
        
        return y_pred_test, y_pred_train
        
    def svm(self, X_train, y_train, X_test, y_test): 
       
        clf = svm.SVC(decision_function_shape='ovr')
        clf.fit(X_train, y_train)
        
        # Get Cross Validation Score
        # scores = cross_val_score(clf, X_train, y_train, cv=5)
        # print("Mean cross-validation score: %.2f" % scores.mean())
        
        # Get K-Validation Score
        # kfold = KFold(n_splits=10, shuffle=True)
        # kf_cv_scores = cross_val_score(clf, X_train, y_train, cv=kfold )
        # print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
        
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        # accuracy = accuracy_score(y_pred, y_test)
        # print("Accuracy score: %.2f" % accuracy)
        
        return y_pred_test, y_pred_train
    
    def mlp(self, X_train, y_train, X_test, y_test): # just using built-ins for MLP
        
        clf = MLPClassifier(max_iter=300)
        clf.fit(X_train, y_train)
        
        # Get Cross Validation Score
        # scores = cross_val_score(clf, X_train, y_train, cv=5)
        # print("Mean cross-validation score: %.2f" % scores.mean())
        
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        
        return y_pred_test, y_pred_train
    
    
    # Trying to make my own custom mlp... tough!
    def mlp_custom(self, X_train, y_train, X_test, y_test, hidden_dim, num_epochs):
        

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
        
        y_train_reshaped = torch.zeros(list(X_train.size())[0],3)
        y_train_reshaped[:,0] = y_train
        y_train_reshaped[:,1] = y_train
        y_train_reshaped[:,2] = y_train
        
        #print(X_train.shape)
        batch_size = list(X_test.size())[0] # 1 batch is size of test data
        train_feats = list(X_train.size())[1]
        train_size = list(X_train.size())[0]
        input_dim = train_feats #(batch_size, train_feats)
        output_dim = 3 # output classification for each data point in batch
        print('Input Dim: ',input_dim)
        
        my_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim))
        
        optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
        loss_fcn = nn.CrossEntropyLoss()
        
        y_pred_train = torch.zeros(train_size,3)
        y_pred_test = torch.zeros(batch_size,3)
        
        my_model.train()
        
        # Train MLP and once done, predict outputs for X_test
        for i in range(num_epochs):
            
            with torch.autograd.set_detect_anomaly(True):
                
                for ii in range(round(train_size/batch_size)):
                    
                        y_pred_train[ii*batch_size:(ii+1)*batch_size] = my_model(X_train[ii*batch_size:(ii+1)*batch_size, :])
                        
                print('y_pred_train: ',y_pred_train.size())     
                print('y_train: ',y_train.size())  
                
                loss = loss_fcn(y_pred_train, y_train_reshaped)
                optimizer.zero_grad()
                loss.backward(retain_graph = True) # retain_graph = True
                optimizer.step()
        
    
        my_model.eval()
        
        # Feed test through trained model to get test predictions
        y_pred_test = my_model(X_test)
        y_pred_test = y_pred_test.argmax(dim=1)
               
        
        # Convert both back to numpy
        y_pred_test = y_pred_test.detach().numpy()
        #y_pred_train = y_pred_test.detach().numpy()
        
        return np.rint(y_pred_test) #, np.rint(y_pred_train)
  

        
        # Train MLP and once done, predict outputs for X_test
        for i in range(num_epochs):
            
            with torch.autograd.set_detect_anomaly(True):
            
                for ii in range(batch_size):
                    y_pred_train[ii] = my_model(X_train[ii, :])
                    #y_pred_train[ii] = y_pred_train[ii].clone()
                
                loss = loss_fcn(y_pred_train, y_train)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        
        my_model.eval()
        
        # Feed test through trained model to get test predictions
        for i in range(num_epochs):
            
            for ii in range(list(X_test.size())[0]):
                y_pred_test[ii] = my_model(X_test[ii, :])
               
        
        # Convert both back to numpy
        y_pred_test = y_pred_test.detach().numpy()
        y_pred_train = y_pred_test.detach().numpy()
        
        return np.rint(y_pred_test), np.rint(y_pred_train)
  
    def vr(self, top_n): # variance ratio filter
    
        normalized_data = self.normalized_data    
    
        # Adjusting AVR Code to Make VR
        train_data = normalized_data[:, :-1]
        train_label = normalized_data[:, -1]
        
        # Get list of label names to iterate through and find number of instances per label
        unique, counts = np.unique(train_label, return_counts=True)
        unique = unique.astype(int)
        label_histogram = {label:count for (label, count) in zip(unique, counts)}
            
        # Populate C x M within class variance matrix
        within_class_var = np.zeros((len(unique), train_data.shape[1])) # instantiate C x M matrix where C is number of classes, M is features
            
        for k in unique:
            
            where_k = np.where(train_label == k)[0] # where_k is indices of class k in vector train_label
            k_data = train_data[where_k[:,], :] # select data points with label k and each of their features returns (where_k x M)
            within_class_var_k = np.atleast_2d(np.var(k_data, axis=0)) # Take variance along where_k dimension, returns (1 x M)
            within_class_var[k, :] = within_class_var_k
            
        # Compute Variance Ratio (1 x M)
            
        cross_class_var = np.var(train_data, axis=0) # computes variance along N dimension in N x M, returning 1 x M
        vr = np.zeros((train_data.shape[1],))
                
        for f in range(train_data.shape[1]): # iterate through each feature
                
            denominator_classes_sum = 0 # keep running sum of denominator for each class
                
            # Iterate through each class in list of classes
            for c in unique:
                
                denominator_classes_sum += within_class_var[c, f]
    
                
            full_denominator = denominator_classes_sum/(len(unique)) # 1/C * denominator_classes_sum
            vr[f] = cross_class_var[f]/full_denominator
            
        # Sort vr in descending order and record indices and scores
        filter_inds = np.flip(np.argsort(vr)) # sort from highest to lowest
        filter_scores = vr[filter_inds] # get score corresponding for each index
        
        filter_inds = filter_inds[:top_n]
        filter_scores = filter_scores[:top_n]
        
        return filter_inds #, filter_scores
    
    
    def avr(self, top_n): # augmented variance ratio filter
    
        normalized_data = self.normalized_data       
    
        # Adopting AVR code from Project 2
        train_data = normalized_data[:, :-1]
        train_label = normalized_data[:, -1]
        
        # Get list of label names to iterate through and find number of instances per label
        unique, counts = np.unique(train_label, return_counts=True)
        unique = unique.astype(int)
        label_histogram = {label:count for (label, count) in zip(unique, counts)}
            
        # Populate C x M within class variance and within class mean matrices
        within_class_var = np.zeros((len(unique), train_data.shape[1])) # instantiate C x M matrix where C is number of classes, M is features
        within_class_mean = np.zeros((len(unique), train_data.shape[1]))
            
        for k in unique:
            
            where_k = np.where(train_label == k)[0] # where_k is indices of class k in vector train_label
            k_data = train_data[where_k[:,], :] # select data points with label k and each of their features returns (where_k x M)
            within_class_mean_k = np.atleast_2d(np.mean(k_data, axis=0)) # Take mean along where_k dimension, returns (1 x M)
            within_class_var_k = np.atleast_2d(np.var(k_data, axis=0)) # Take variance along where_k dimension, returns (1 x M)
            within_class_mean[k, :] = within_class_mean_k
            within_class_var[k, :] = within_class_var_k
            
        # Compute Augmented Variance Ratio (1 x M)
            
        cross_class_var = np.var(train_data, axis=0) # computes variance along N dimension in N x M, returning 1 x M
        avr = np.zeros((train_data.shape[1],))
                
        for f in range(train_data.shape[1]): # iterate through each feature
                
            denominator_classes_sum = 0 # keep running sum of denominator for each class
                
                # Iterate through each class in list of classes
            for c in unique:
                    
                unique_copy = unique
                c_idx = np.where(unique_copy == c)
                unique_copy = np.delete(unique_copy, c_idx) # remove c from list of classes
                current_min = 1000000000000000000000 # initialize first min to be very large so whatever follows will be less
                    
                for m in unique_copy: # iterate through list of classes with class c removed from list 
                        
                    test_diff = within_class_mean[c, f] - within_class_mean[m, f]
                    test_min = np.abs(test_diff)
                        
                    if test_min < current_min: # find min difference of within class mean where i != j
                            
                        current_min = test_min
                        current_min_idx = m
                    
                    min_ij = current_min
                    
                    # Only compute running sum if min_ij is not 0, otherwise infinity problem
                    if min_ij > 0:
                        denominator_classes_sum += within_class_var[c, f]/min_ij
                
                full_denominator = denominator_classes_sum/(len(unique)) # 1/C * denominator_classes_sum
                avr[f] = cross_class_var[f]/full_denominator
            
            # Sort avr in descending order and record indices and scores
        filter_inds = np.flip(np.argsort(avr)) # sort from highest to lowest
        filter_scores = avr[filter_inds] # get score corresponding for each index
        
        filter_inds = filter_inds[:top_n]
        filter_scores = filter_scores[:top_n]
        
        return filter_inds #, filter_scores
    
    # Referenced https://www.askpython.com/python/examples/principal-component-analysis
    def pca_reduction(self, top_n): # take Z normalized data and perform PCA on it
    
        normalized_data = self.normalized_data   
        
        covariance_mat = np.cov(normalized_data[:, :-1], rowvar = False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigval = eigenvalues[sorted_idx]
        sorted_eigvect = eigenvectors[:, sorted_idx]
        eig_vect_reduced = sorted_eigvect[:, 0:top_n].real # all imag parts are 0 anyway
        X_reduced = np.dot(normalized_data[:, :-1], eig_vect_reduced) # Reduced transforms N x F original data by multiplying by F x num_comps
        
        return X_reduced # in form N x num_comps
    
    
    def correlation_reduction(self): 
        
        normalized_data = self.normalized_data   
        
        # Calculate correlation matrix from z normalized data
        covariance_mat = np.cov(normalized_data[:, :-1], rowvar = False) # covariance between every combination of features (F x F)
        diag = np.sqrt(np.diag(covariance_mat)) # Get sqrt of covariance matrix diagonal (F,1)
        outer_prod = np.outer(diag, diag) # returns F x F matrix populated with value on diagonal of covariance matrix
        correlation_mat = covariance_mat / outer_prod
        correlation_mat[covariance_mat == 0] = 0
        
        # Iterate through features of correlation matrix and eliminate feature if abs(correlation) > 0.05
        feats_selected = []
        
        for col in range(correlation_mat.shape[1]):
            
            strong_correl_detected = False
            
            for row in range(correlation_mat.shape[0]):
                
                if np.abs(correlation_mat[row, col]) > 0.2 and row != col: # if correlation not negligible (> 0.2), don't select
                    
                    strong_correl_detected = True
            
            if not strong_correl_detected: # if no strong correlation detected, add feat
                
                feats_selected.append(col)
        
        feats_selected = np.array(feats_selected)
                    
            
        return feats_selected #, correlation_mat
    
    # Use open source library from github to perform minimum redundancy maximum relevance feature selection
    # https://github.com/smazzanti/mrmr
    def mRMR(self, top_n): 
        
        normalized_data = self.normalized_data   
        
        # split data into training and test sets 
        
        y = normalized_data[:, -1] # select last column
        X = normalized_data[:, :-1] # select all columns removing last
        
        # Convert training data and labels to pandas dataframe for compatibility with mrmr_classif function
        df_y = pd.DataFrame(y)
        df_X = pd.DataFrame(X)
        
        # Call mrmr_classif and convert output back to numpy array
        selected_features = mrmr_classif(X=df_X, y=df_y, K= top_n) # takes pandas dataframe
        selected_features = np.array(selected_features)
        
        return selected_features

    def sci_py_select(self, top_n):
        
        normalized_data = self.normalized_data 
        # Select some sci-py feature reduction technique in future...

