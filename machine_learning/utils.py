# Plotting functions and data loading for model configuration testing

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Manual Plotting
def plot_bagging():
    X = ['XG/MRMR','XG/VR','XG/AVR','CAT/Corr', 'CAT/MRMR','CAT/AVR','KNN/MRMR', 'KNN/VR']
    accuracy = [54.6, 51.9, 52.9, 51.7, 53.7, 53.3, 51.2, 49.6]
    false_pos = [36.8, 39, 38.1, 37, 36.8, 38.9, 37.3, 41.2]
      
    X_axis = np.arange(len(X))
      
    plt.bar(X_axis - 0.2, accuracy, 0.4, label = 'Accuracy')
    plt.bar(X_axis + 0.2, false_pos, 0.4, label = 'False Positive')
    
    plt.xticks(X_axis, X, rotation=25)
    plt.xlabel("Classifier + Filter")
    plt.ylabel("Percentage (%)")
    plt.title("Bagging Classifier/Filter Pairs")
    plt.legend()
    plt.show()

def plot_stacking():
    X = ['Reg/MRMR','Reg/AVR','Reg/VR','SVM/MRMR', 'SVM/AVR','SVM/VR','NB/MRMR', 'NB/AVR', 'NB/VR']
    accuracy = [53.8, 52.2, 49.9, 53.5, 49.7, 49.7, 48.3, 47.2, 43.5]
    false_pos = [36.8, 37.2, 40.4, 36.8, 40, 40.9, 36.8, 38.7, 39.8]
      
    X_axis = np.arange(len(X))
      
    plt.bar(X_axis - 0.2, accuracy, 0.4, label = 'Accuracy')
    plt.bar(X_axis + 0.2, false_pos, 0.4, label = 'False Positive')
    
    plt.xticks(X_axis, X, rotation=25)
    plt.xlabel("Level 1 Classifier + Filter")
    plt.ylabel("Percentage (%)")
    plt.title("Stacking Classifier/Filter Pairs")
    plt.legend()
    plt.show()

# Automatic plotting
def plot_auto(plot_dict, chdir, ensemble_method): 
    os.chdir(chdir)
    
    if ensemble_method == 'bagging':
        title_str = "Bagging Classifier/Filter Pairs"
    
    elif ensemble_method == 'stacking':
        title_str = "Stacking Classifier/Filter Pairs"
    
    else:
        raise Exception('Wrong ensemble method string entered')
        sys.exit(1)
    
    names = []
    accuracy = []
    acc_sem = []
    false_pos = []
    false_pos_sem = []
    
    for key in plot_dict.keys():
        
        name = key.split('.')[0]
        names.append(name)
        accuracy.append(100*plot_dict[key][0]) # convert decimal to percentage 
        acc_sem.append(100*plot_dict[key][1])
        false_pos.append(100*plot_dict[key][2])
        false_pos_sem.append(100*plot_dict[key][3])
        
    # print('Names: ', len(names))
    # print('Accuracy: ', len(accuracy))
    # print('Accuracy SEM: ', len(acc_sem))
    # print('False Positive Rate: ', len(false_pos))
    # print('False Positive SEM: ', len(false_pos_sem))
    
    if ensemble_method == 'bagging':
        X = ['CAT/AVR','CAT/Corr','CAT/MRMR','KNN/MRMR', 'KNN/VR', 'XG/AVR', 'XG/MRMR', 'XG/VR']
        
    else:
        X = ['NB/AVR', 'NB/MRMR', 'NB/VR', 'Reg/AVR','Reg/MRMR','Reg/VR', 'SVM/AVR','SVM/MRMR', 'SVM/VR']
      
    X_axis = np.arange(len(X))
      
    plt.bar(X_axis - 0.2, accuracy, 0.4, label = 'Accuracy')
    plt.errorbar(X_axis - 0.2, accuracy, acc_sem, fmt='.', color='Black', elinewidth=2)
    plt.bar(X_axis + 0.2, false_pos, 0.4, label = 'False Positive')
    plt.errorbar(X_axis + 0.2, false_pos, false_pos_sem, fmt='.', color='Black', elinewidth=2)
    
    plt.xticks(X_axis, X, rotation=25)
    plt.xlabel("Classifier + Filter")
    plt.ylabel("Percentage (%)")
    plt.title(title_str)
    plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize="6")
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    savestr = ensemble_method + '-' + dt_string
    plt.savefig(savestr)
    plt.clf() # clear figure to avoid overlapping
    os.chdir('..')
    #plt.show()

# Load data automatically by directory
def load_npz(dir_str):
    
    os.chdir(dir_str)
    plot_dict = {}
    
    for file in os.listdir():
        
        if file.endswith(".npz"):
            
            data = np.load(file, mmap_mode=None, allow_pickle=True)
            plot_dict[file] = [data['avg_acc'].item(), data['avg_acc_sem'].item(), data['false_pos'].item(), data['false_pos_sem'].item()]
            print('Filename: ', file)
            print('Avg Acc: ', data['avg_acc'])
            print('Avg Acc SEM: ', data['avg_acc_sem'])
            print('False Positive Rate: ', data['false_pos'])
            print('False Positive SEM: ', data['false_pos_sem'])
    
    os.chdir('..')
    return plot_dict


            
            
            
            
            
            
            
            
            