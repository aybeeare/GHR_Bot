# Main script for testing ensemble techniques

from GHR_Bot import *
import operator
import os
import numpy as np
import pandas as pd
from sklearn import metrics

# Import user defined functions and classes
from ensemble_techniques import *
from utils import *

def test_bagging(classifiers, filters, its, bagging_its):
    # model: string indicating model to test: e.g. cat, xg...
    # filt: string indicating filter to test: avr, mrmr, vr...
    # its: number of iterations to perform
    
    if not os.path.isdir('bagging_plots'):
        os.mkdir('bagging_plots')
    
    for model, filt in zip(classifiers, filters):

        sum_conf = np.zeros((3,3))
        divider = its*np.ones((3,3))
        conf_matrices = []
        
        for i in range(its):
            conf_matrix, recs_list = bagging(model, filt, bagging_its, True, num_feats=20) # bagging at 15 good, bagging at 10?
            conf_matrices.append(conf_matrix)
            sum_conf += conf_matrix

            # Count up number of times tick is recommended in recs_list (taking the ensemble of the ensemble idea :) )
            if i == 0:
                tick_tally = dict(zip(recs_list, [0]*len(recs_list)))
            
            else: # If tick in recs_list in dict, increment, else, instantiate with value 0.

                for new_rec in recs_list:

                    if new_rec in tick_tally.keys():
                        tick_tally[new_rec] += 1
                    else:
                        tick_tally[new_rec] = 1
        
        tick_tally.update((x, y/its) for x, y in tick_tally.items())
        tick_tally = sorted(tick_tally.items(),key=operator.itemgetter(1),reverse=True)
        print('Dictionary Tick...', tick_tally)

        avg_conf_matrix = np.divide(sum_conf, divider)
        
        # Compute std
        sum_squared = np.zeros((3,3))
        for conf in conf_matrices:
            sum_squared += np.square(np.subtract(conf,avg_conf_matrix))
            
        std_conf_matrix = np.sqrt(np.divide(sum_squared, divider)) # divide each element in sum squared by # of samples and take square root
        std_error_mean = np.divide(std_conf_matrix, np.sqrt(divider)) # compute standard error of the mean (std/sqrt(N))
        
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
        
        os.chdir('bagging_plots') # change into save directory
        save_str = str(model) + '_' + str(filt) + '.png'
        npy_str = str(model) + '_' + str(filt)
        plt.savefig(save_str)
        plt.clf()
        np.savez(npy_str, avg_acc=avg_acc_norm, avg_acc_sem=avg_acc_sem, false_pos=false_pos, false_pos_sem=false_pos_sem)
        os.chdir('..')

def test_stacking(filters, its, svms, nbs):
    # filt: string indicating filter to test: avr, mrmr, vr...
    # its: number of iterations to perform
    
    if not os.path.isdir('stacking_plots'):
        os.mkdir('stacking_plots')
    
    for filt, use_svm, use_nb in zip(filters, svms, nbs):
    
        sum_conf = np.zeros((3,3))
        divider = its*np.ones((3,3))
        conf_matrices = []
        
        for i in range(its):
            conf_matrix = stacking({"xg": 6, "cat": 6, "svm": 5, "knn": 5, "mlp": 5}, filt, use_svm, use_nb) 
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
        
        os.chdir('stacking_plots') # change into save directory
        
        if use_svm == False and use_nb == False:
            model = 'reg'
        elif use_svm == True and use_nb == False:
            model = 'svm'
        elif use_svm == False and use_nb == True:
            model = 'bayes'
        else:
            model = 'Wrong_Booleans'
            
        save_str = str(model) + '_' + str(filt) + '.png'
        npy_str = str(model) + '_' + str(filt)
        plt.savefig(save_str)
        plt.clf()
        np.savez(npy_str, avg_acc=avg_acc_norm, avg_acc_sem=avg_acc_sem, false_pos=false_pos, false_pos_sem=false_pos_sem)
        os.chdir('..')
       

def main():
    
    # Calling various functions to collect data...
    
    # classifiers = ['xg', 'xg', 'xg', 'cat', 'cat', 'cat', 'knn', 'knn'] 
    # filters = ['mrmr', 'vr', 'avr', 'correlation', 'mrmr', 'avr', 'mrmr', 'vr']

    fetch_current(0) # Run GHR Bot to Fetch sibs this month (and spit out preprocessing)
    classifiers = ['xg'] 
    filters = ['mrmr']
    
    test_bagging(classifiers, filters, its = 100, bagging_its = 40)
    
    # filters = ['mrmr', 'avr', 'vr', 'mrmr', 'avr', 'vr', 'mrmr', 'avr', 'vr']
    # svms = [False, False, False, True, True, True, False, False, False]
    # nbs = [False, False, False, False, False, False, True, True, True]
    
    # test_stacking_builtins(filters, 10, svms, nbs)
    
    # test_stacking(filters, 10, svms, nbs) # if use_svm and use_nb both false, defaults to linear regression
    # dir_str1 = 'stacking_plots_z_norm'
    # plot_stack = load_npz(dir_str1) # plot bag is a dict with npz files and the corresponding accuracy/sem and false positive/sem
    # plot_auto(plot_stack, dir_str1, ensemble_method = 'stacking')
    
    # dir_str2 = 'bagging_plots_min_max_norm'
    # plot_bag = load_npz(dir_str2) # plot bag is a dict with npz files and the corresponding accuracy/sem and false positive/sem
    # plot_auto(plot_bag, dir_str2, ensemble_method = 'bagging')
    #print('Bagging Data: ', plot_bag)
    
    # dirs = ['bagging_plots_z_norm','stacking_plots_z_norm', 'bagging_plots_min_max_norm','stacking_plots_min_max_norm']
    
    # for dir_str in dirs:
    #     print('##########################')
    #     print('Directory Name: ', dir_str)
    #     print('##########################')
    #     my_dict = load_npz(dir_str)
    #     print('\n')
    
    pass
    
    

if __name__ == '__main__':
    main()


