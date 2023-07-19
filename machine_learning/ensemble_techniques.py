# Preprocessing Functions to Try

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from mrmr import mrmr_classif
import matplotlib.pyplot as plt

# Import user defined functions and classes
from preprocessing import *
from classes import Classifier   
    
    
#def main():


def stacking(classifier_dict, filt, use_svm, use_nb, pred_newbuys): #classifiers_list, filters):
    
    # classifiers: dict with string for classifier name as key and number of votes in ensemble as value
    # filt: string of filter to apply
    
    # Convert classifiers dictionary to classifier list
    classifiers = []
    
    for key in classifier_dict.keys():
        
        voters = classifier_dict[key]
        
        for i in range(voters):
            
            classifiers.append(key)
    
    unnormalized_data = train_cleanup()
    normalized = z_normalization(unnormalized_data)  
    #normalized = min_max_normalization(unnormalized_data)
        
    feat_select_filter = False
    
    classifier = Classifier(normalized)
    
    # Select correct filter method and return filtered features
    if filt == 'avr':
        feats_selected = classifier.avr(top_n = 25)
        feat_select_filter = True
                  
    elif filt == 'vr':
        feats_selected  = classifier.vr(top_n = 25)
        feat_select_filter = True    
        
    elif filt == 'mrmr':
        feats_selected  = classifier.mRMR(top_n = 25)
        feat_select_filter = True
         
    elif filt == 'pca':
        X = classifier.pca_reduction(top_n = 9)
        
    elif filt == 'correlation':
        feats_selected  = classifier.correlation_reduction()
        feat_select_filter = True
        
    else: # wrong filter specified
        raise Exception('Wrong filter specified')
        
    
    if feat_select_filter: # if method returns feature indices
    
        X = normalized[:, feats_selected] # select all columns removing last
        
    y = normalized[:, -1] # select last column
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) # random state = 0for repeatability, get filtered X_test and y_test to test classifier
    
    y_preds_train = np.zeros((y_train.shape[0], len(classifiers))) # size for 20% test split
    y_preds_test = np.zeros((y_test.shape[0], len(classifiers)))
    
    i = 0 # populate matrices above by column index
    
    for classifier_type in classifiers: 
                
        # Select correct classifier method and pass dimension-reduced dataset and labels (do sampling with replacement splitting, how to tally votes for different samples?? )
        if classifier_type == 'xg':
            y_pred_test, y_pred_train = classifier.xg_boost(X_train, y_train, X_test, y_test)
            
        elif classifier_type == 'cat':
            y_pred_test, y_pred_train = classifier.cat_boost(X_train, y_train, X_test, y_test)
            y_pred_test = y_pred_test.flatten()
            y_pred_train = y_pred_train.flatten()
            
        elif classifier_type == 'knn':
            k_neighbors_dict = {'vr': 9, 'mrmr': 10, 'avr': 11, 'correlation': 10, 'pca': 8} # optimal k_neighbors for each filter type
            k_neighbors = k_neighbors_dict[filt]
            y_pred_test, y_pred_train = classifier.knn(X_train, y_train, X_test, y_test, k_neighbors)
            
        elif classifier_type == 'svm': # svm
            y_pred_test, y_pred_train = classifier.svm(X_train, y_train, X_test, y_test)
        
        elif classifier_type == 'mlp': # mlp
            y_pred_test, y_pred_train = classifier.mlp(X_train, y_train, X_test, y_test)
        
        else: # wrong classifier specified
            print('Wrong classifier specified: ', classifier_type)
            continue
    
        y_preds_train[:, i] = y_pred_train 
        y_preds_test[:, i] = y_pred_test 
        
        i += 1
    
    # Train Logistic Regression Classifier with predicted outputs from classifiers using training data as input
    logistic_reg = LogisticRegression()
    logistic_reg.fit(y_preds_train, y_train)
    # https://www.analyticsvidhya.com/blog/2021/08/ensemble-stacking-for-machine-learning-and-deep-learning/
    
    # Feed test data through classifiers and feed outputs through regression classifier to get predictions
    y_pred = logistic_reg.predict(y_preds_test)
    
    if use_svm: # Use SVM instead of LogisticRegression for final classifier
        
        clf = svm.SVC(decision_function_shape='ovr') 
        clf.fit(y_preds_train, y_train)
        y_pred = clf.predict(y_preds_test)
    
    if use_nb:
        
        GNBclf = GaussianNB()
        GNBclf.fit(y_preds_train, y_train)
        y_pred = GNBclf.predict(y_preds_test)
        
    # Predictions of each classifier is a feature, feed to linear regression model for final prediction
    # np.savetxt("answers.csv", y_test, delimiter=",")
    # np.savetxt("votes.csv", y_preds_test, delimiter=",")
    # accuracy = accuracy_score(y_pred, y_test)
    # print("Accuracy score: %.2f" % accuracy)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    # cm_display.plot()
    # plt.show()
    return confusion_matrix


def bagging(classifier_type, filt, iterations, pred_newbuys, num_feats): #classifiers_list, filters):
    
    # classifiers: List of strings naming classifiers to test
    # filt: string of filter to apply
    
    unnormalized_data = train_cleanup()
    unnormalized_newbuys, test_syms = new_buys_format()
    normalized = z_normalization(unnormalized_data)
    #normalized = min_max_normalization(unnormalized_data)
    normalized_newbuys = z_normalization(unnormalized_newbuys)
    #normalized_newbuys = min_max_normalization(unnormalized_newbuys)
     
    # Figure out how to initialize zeros matrix with correct size (len(y_pred), len(classifiers)*len(filters)), don't know len(y_pred) till after split
        
    
    feat_select_filter = False
    
    classifier = Classifier(normalized)
    
    # Select correct filter method and return filtered features
    if filt == 'avr':
        feats_selected = classifier.avr(top_n = 25)
        feat_select_filter = True
                  
    elif filt == 'vr':
        feats_selected  = classifier.vr(top_n = 25)
        feat_select_filter = True    
        
    elif filt == 'mrmr':
        feats_selected  = classifier.mRMR(top_n = 25) # tried 40, got some okay results
        feat_select_filter = True
    
    elif filt == 'correlation':
        feats_selected  = classifier.correlation_reduction()
        num_feats = len(feats_selected)
        feat_select_filter = True
         
    elif filt == 'pca':
        X = classifier.pca_reduction(top_n = 9)
    
    elif filt == 'na': # no filter selected
        X = normalized[:, :-1]
        
    else: # wrong filter specified
        raise Exception('Wrong filter specified')

    X = normalized[:, feats_selected] # select all columns removing last
    y = normalized[:, -1] # select last column
    X_training_set, X_test, y_training_set, y_test = train_test_split(X, y, test_size=0.05) # random_state = 0 for repeatability, get filtered X_test and y_test to test classifier
    y_preds = np.zeros((X_test.shape[0], iterations)) # size for 20% test split
        
    if feat_select_filter and pred_newbuys:

        X_test = normalized_newbuys[:, feats_selected] # using new buys as test set 
        y_test = np.zeros((X_test.shape[0],)) # dummy y_test, unused in this case since pred_newbuys have no labels. Just format fcn. inputs
        y_preds = np.zeros((X_test.shape[0], iterations)) # prediction for each new buy  

    i = 0 # populate matrices above by column index

    for its in range(iterations):
        
        # Select random set of features from training set and perform split
        X_train, X_test_notused, y_train, y_test_notused = train_test_split(X_training_set, y_training_set, test_size=0.2, shuffle = True) # for each classifier, do random split for training
        
        if feat_select_filter:
            rand_feats = np.random.randint(0, high= len(feats_selected), size = num_feats)
            X_train_reduced = X_train[:, rand_feats]
            X_test_reduced = X_test[:, rand_feats]
        
        else: # PCA case
            X_train_reduced = X_train
            X_test_reduced = X_test
        
        
        # Select correct classifier method and pass dimension-reduced dataset and labels (do sampling with replacement splitting, how to tally votes for different samples?? )
        if classifier_type == 'xg':
            y_pred, y_pred_train = classifier.xg_boost(X_train_reduced, y_train, X_test_reduced, y_test)
            
        elif classifier_type == 'cat':
            y_pred, y_pred_train = classifier.cat_boost(X_train_reduced, y_train, X_test_reduced, y_test)
            y_pred = y_pred.flatten()
            y_pred_train = y_pred_train.flatten()
            
        elif classifier_type == 'knn':
            k_neighbors_dict = {'vr': 9, 'mrmr': 10, 'avr': 11, 'correlation': 10, 'pca': 8, 'na': 8} # optimal k_neighbors for each filter type
            k_neighbors = k_neighbors_dict[filt]
            y_pred, y_pred_train = classifier.knn(X_train_reduced, y_train, X_test_reduced, y_test, k_neighbors)
            
        elif classifier_type == 'svm': # svm
            y_pred, y_pred_train = classifier.svm(X_train_reduced, y_train, X_test_reduced, y_test)
        
        elif classifier_type == 'mlp': # mlp
            y_pred, y_pred_train = classifier.mlp(X_train, y_train, X_test, y_test)
        
        else: # wrong classifier specified
            raise Exception('Wrong classifier specified')
    
        y_preds[:, i] = y_pred 
        
        i += 1

    # Figure out committee vote by taking the row-wise mode of the preds matrix, compare resulting array with y_tests (the true values for the preds)
    #np.savetxt("foo.csv", y_preds, delimiter=",")
    y_pred = stats.mode(y_preds, axis=1)
    y_pred_list = []
    for x in y_pred[0]:
        y_pred_list.append(int(x[0]))
    y_pred = np.array(y_pred_list)

    if pred_newbuys:
        newbuys_labeled = np.concatenate((np.atleast_2d(np.array(test_syms)).T, unnormalized_newbuys, np.atleast_2d(y_pred).T), axis=1)
        newbuys_recs_df = pd.DataFrame(newbuys_labeled)
        # print('New Buys Recs Dataframe:')
        # print(newbuys_recs_df)
        # newbuys_recs_df.to_csv('newbuys_bagging_recs.csv')

        # Identify Strong Buys, Store in DF, write to csv...

        recs_list = []

        for sym, pred in zip(test_syms, y_pred_list):

            if pred == 2 and sym not in recs_list:

                recs_list.append(sym)
        
        print('Recs List: ', len(recs_list))
        pd.DataFrame(np.array(recs_list)).to_csv('Rec_Tickers.csv')
        confusion_matrix = np.zeros((3,3))
        return confusion_matrix, recs_list



    #print('y_pred list: ', y_pred)
    #print('Shape of y_pred', y_pred[0])
    #print('Shape of y_test', y_test)
    #accuracy = accuracy_score(y_pred, y_test)
    #print("Accuracy score: %.2f" % accuracy)

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    return confusion_matrix, y_pred_list
    #print('Confusion Matrix: ', confusion_matrix)
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    # cm_display.plot()
    # plt.show()
    

# Used built in functions to do stacking to validate my from scratch function, results same :)
def stacking_with_built_ins(filt, use_svm, use_nb):
    
    # Followed: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
    
    unnormalized_data = train_cleanup()
    normalized = z_normalization(unnormalized_data)  
    #normalized = min_max_normalization(unnormalized_data)  
    # Figure out how to initialize zeros matrix with correct size (len(y_pred), len(classifiers)*len(filters)), don't know len(y_pred) till after split
        
    feat_select_filter = False
    
    classifier = Classifier(normalized)
    
    # Select correct filter method and return filtered features
    if filt == 'avr':
        feats_selected = classifier.avr(top_n = 19)
        feat_select_filter = True
                  
    elif filt == 'vr':
        feats_selected  = classifier.vr(top_n = 25)
        feat_select_filter = True    
        
    elif filt == 'mrmr':
        feats_selected  = classifier.mRMR(top_n = 25)
        feat_select_filter = True
    
    elif filt == 'correlation':
        feats_selected  = classifier.correlation_reduction()
        feat_select_filter = True
         
    elif filt == 'pca':
        X = classifier.pca_reduction(top_n = 9)
    
    elif filt == 'na': # no filter selected
        X = normalized
        
    else: # wrong filter specified
        raise Exception('Wrong filter specified')
        
    
    if feat_select_filter: # if method returns feature indices
    
        X = normalized[:, feats_selected] # select all columns removing last
        
    y = normalized[:, -1] # select last column
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # random_state = 0 for repeatability, get filtered X_test and y_test to test classifier
    
    
    # define the base models
    level0 = list()
    level0.append(('cat', CatBoostClassifier(
                           loss_function='MultiClass',
                           verbose=True)))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('svm', svm.SVC(decision_function_shape='ovr')))
    level0.append(('xg', XGBClassifier(objective = 'multi:softmax', num_class = 3)))
    level0.append(('mlp', MLPClassifier(max_iter=300)))
    
    # define meta learner model
    if use_svm == False and use_nb == False:
        level1 = LogisticRegression()
    
    elif use_svm == True and use_nb == False:
        level1 = svm.SVC(decision_function_shape='ovr') 
    
    elif use_svm == False and use_nb == True:
        level1 = GaussianNB()
    
    else:
        raise Exception('Wrong level 1 classifier specified')
        
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
         
    # Evaluate model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_pred, y_test)
    # print("Accuracy score: %.2f" % accuracy)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    # cm_display.plot()
    # plt.show()

    return confusion_matrix

# Just used built-in stacking function for sanity check, got very similar results to mine so should be good...
def test_stacking_builtins(filters, its, svms, nbs):
    # filt: string indicating filter to test: avr, mrmr, vr...
    # its: number of iterations to perform

    if not os.path.isdir('stacking_builtins_plots'):
        os.mkdir('stacking_builtins_plots')
    
    for filt, use_svm, use_nb in zip(filters, svms, nbs):
    
        sum_conf = np.zeros((3,3))
        divider = its*np.ones((3,3))
        conf_matrices = []
        
        if use_svm == False and use_nb == False:
            model = 'reg'
        elif use_svm == True and use_nb == False:
            model = 'svm'
        elif use_svm == False and use_nb == True:
            model = 'bayes'
        else:
            model = 'Wrong_Booleans'
        
        for i in range(its):
            conf_matrix = stacking_with_built_ins(filt, use_svm, use_nb)
            conf_matrices.append(conf_matrix)
            sum_conf += conf_matrix
        
        avg_conf_matrix = np.divide(sum_conf, divider)
        
        # Compute std
        sum_squared = np.zeros((3,3))
        for conf in conf_matrices:
            sum_squared += np.square(np.subtract(conf,avg_conf_matrix))
            
        std_conf_matrix = np.sqrt(np.divide(sum_squared, divider)) # divide each element in sum squared by its and take square root
        std_error_mean = np.divide(std_conf_matrix, np.sqrt(divider))
        
        avg_accuracy = np.trace(avg_conf_matrix) # sum along diagonal of average conf. matrix is avg. accuracy
        avg_accuracy_sem = np.trace(std_error_mean)
        avg_acc_norm = avg_accuracy/419
        avg_acc_sem = avg_accuracy_sem/419
        false_pos = (avg_conf_matrix[0,2])/(avg_conf_matrix[2,2] + avg_conf_matrix[1,2] + avg_conf_matrix[0,2])
        false_pos_sem = std_error_mean[0,2]/419
        
        tot = 419*np.ones((3,3))
        avg_conf_normalized = 100*np.divide(avg_conf_matrix, tot) # display in %
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = avg_conf_normalized)
        cm_display.plot()
        
        os.chdir('stacking_builtins_plots') # change into save directory
            
        save_str = str(model) + '_' + str(filt) + '.png'
        npy_str = str(model) + '_' + str(filt)
        plt.savefig(save_str)
        plt.clf()
        np.savez(npy_str, avg_acc=avg_acc_norm, avg_acc_sem=avg_acc_sem, false_pos=false_pos, false_pos_sem=false_pos_sem)
        os.chdir('..')










# Scratch Paper/Code Below...

# X_reduced = pca_reduction(normalized_data, 9) # Reduced to 9 dim performed best, 8 nearest neighbors was also best.

# feats_chosen = []
# filter_inds, correl_mat = correlation_reduction(normalized_data)
# # print('Correlation Reduction Inds: ', filter_inds)
# feats_chosen.append(filter_inds.tolist())

# filter_inds, filter_scores = avr_filter(normalized_data, 19) # AVR: top 15 yielded 55% accuracy, top 25, 30, 35, yieled 53%, VR: top 25 with 53%
# feats_chosen.append(filter_inds.tolist())
# # print('AVR Inds: ', filter_inds)
# filter_inds, filter_scores = vr_filter(normalized_data, 25) 
# feats_chosen.append(filter_inds.tolist())
# # print('VR Inds: ', filter_inds)
# filter_inds = mRMR(normalized_data, 25)
# feats_chosen.append(filter_inds.tolist())
# # print('mRMR Inds: ', filter_inds)

# result = sum(feats_chosen, [])
# unique = np.unique(np.array(result)).tolist()

# print('Unique Features: ', len(unique))




    #col_var = get_dataset_stats(unnormalized_data)
#xg_boost_test(normalized_data, filter_inds, True, False, X_reduced)
#knn_test(normalized_data, filter_inds, True, False, X_reduced, 10) # AVR: 11 neighbors performed best. VR: 9, mRMR: 10
#cat_boost_test(normalized_data, filter_inds, True, False, X_reduced)
#svm_test(normalized_data, filter_inds, True, False, X_reduced)

# if __name__ == "__main__":
#     main()
        