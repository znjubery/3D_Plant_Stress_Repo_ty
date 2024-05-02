# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:27:43 2023

@author: therin young, therin.young@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt
import imblearn
print(imblearn.__version__)
#import models and performance metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn import svm as svm_linear

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts
import sklearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
import pandas as pd

print(imblearn.__version__)
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import KMeansSMOTE


import dataframe_image as dfi
import os
import argparse
 

def main():
    parser = argparse.ArgumentParser(description="Train ML models on canopy color percentage data to predict IDC scores")
    parser.add_argument("csv_file", help="Path to the CSV file containing point cloud data e.g. filenames, IDC scoring, color percentages")
    parser.add_argument("output_path", help="folder path to store results")

    args = parser.parse_args()
    
    # Read the CSV data file
    df = pd.read_csv(args.csv_file, index_col=False)
    
    df.rename(columns = {'percgreen':'%g','percyellow':'%y','percbrown':'%b','idc_score':'rating'},inplace=True)
    
    #update datatype for rating column to integer 
    df['rating'] = df['rating'].astype(int)
    
    
    #Base Model Data
    #define features (X) and classification labels (y)
    X = np.array(df[['%b','%g','%y']])
    y = np.array(df['rating'])

    u, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv)
    max_count_base = max(counts)
    print(max_count_base)
    
    #Model 0 Data
    bin_lst = []
    for i in range(0,df.shape[0]):
        if df['rating'][i] == 1:
            bin_lst.append(1)
        elif df['rating'][i] == 2:
            bin_lst.append(1)
        elif df['rating'][i] == 3:
            bin_lst.append(2)
        elif df['rating'][i] == 4:
            bin_lst.append(3)
        elif df['rating'][i] == 5:
            bin_lst.append(3)
    
    df['severity_level'] = bin_lst
    
    #define features (X) and classification labels (y)
    X_m0 = np.array(df[['%b','%g','%y']])
    y_m0 = np.array(df['severity_level'])
    
    u, inv = np.unique(y_m0, return_inverse=True)
    counts = np.bincount(inv)
    max_count_m0 = max(counts)
    print(max_count_m0)
    
    
    #Model 1 Data
    #model1 dataset
    X_m1 = X[((y ==1)|(y == 2))]
    y_m1 = y[((y == 1)|(y == 2))]
    
    u, inv = np.unique(y_m1, return_inverse=True)
    counts = np.bincount(inv)
    max_count_m1 = max(counts)
    print(max_count_m1)
    
    #Model 2 Data
    #model2 dataset
    X_m2 = X[((y == 4)|(y == 5))]
    y_m2 = y[((y == 4)|(y == 5))]
    
    u, inv = np.unique(y_m2, return_inverse=True)
    counts = np.bincount(inv)
    max_count_m2 = max(counts)
    print(counts)
    
    #Misclassification Cost
    def misclassification_cost(arg1, arg2):
    
        # function for calculating the missclassification cost of a classifier given test labels and predicted labels returned from the
        # trained classifier.
        
        '''
        inputs:
        arg1 = array of test labels
        arg2 = array of predicted labels
        
        returns:
        misclassification cost
        '''
    
        #print confusion matrix
        CM = confusion_matrix(arg1,arg2)
        print(CM)
    
        #define cost matrix shape
        cM = np.zeros(CM.shape)
    
        #assign weights to cost matrix
        if cM.shape == (3,3):
            cM[0] = [0,1,2]
            cM[1] = [1,0,1]
            cM[2] = [2,1,0]
    
        # for binary classification
        elif cM.shape ==(2,2):
            cM[0] = [0,1]
            cM[1] = [1,0]
    
        elif cM.shape == (5,5):
            cM[0] = [0,1,2,3,4]
            cM[1] = [1,0,1,2,3]
            cM[2] = [2,1,0,1,2]
            cM[3] = [3,2,1,0,1]
            cM[4] = [4,3,2,1,0]
    
        #calculate classification cost
        cM_matrix = np.matrix(CM * cM)
        clcost = cM_matrix.sum()/arg2.shape[0]
        
        return(clcost)
    
    #Add Hierarchical Results
    
    #Calculate performance of hierarchical classifier for unbinned as-is data

    def add_hierarchical_results(arg1, arg2, arg3):
        
        '''
        This is a function that adds classification performance of the 
        hierarchical classifier to either the unbinned or binned classifier data dataframe
        
        arg1: array of test labels
        arg2: array of predicted labels
        arg3: dataframe returned from classification_pipeline0 for hierarchical classifier data to be added to
        '''
        
        value = 'hierarchical'
        if value in list(arg3['Model']):
            print('hierarchical data already exists')
            return(arg3)
        else:       
      
    
            report_dict = classification_report(arg1,arg2,output_dict=True)
            model = 'hierarchical'
            accuracy = report_dict['accuracy']
            mpca = report_dict['macro avg']['recall']
            f1_wt = report_dict['weighted avg']['f1-score']
            cost = misclassification_cost(arg1,arg2)
            unique_predictions = np.unique(arg2)
    
            #Add hierarchical classification results for as-is data to unbinned classification results for as-is data
            table = arg3
            table.loc[len(table.index)] = model,accuracy,mpca,f1_wt,cost,unique_predictions
    
            return (table)
        
        #Base Model
        
        def smote_classifier_base(arg1, arg2, arg3, arg4):
    
            '''
            arg1: Input array (features) - Training data (features)
            arg2: Target array  - Training data (labels)
            arg3: number of items in the majority class (max count)
            arg4: random seed for splitting data
            '''
            
            SMOTE_dict = {'SMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),
                          'BorderlineSMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),        
                          #'ADASYN': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),
                         'SVMSMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42)}
            
            
            #create dataframe for storing classification results
            mb_output = pd.DataFrame()
            
            for key_1 in SMOTE_dict.keys():
                
                print(key_1)
            
            
            
            
                modelDict = {'DecisionTreeClassifier':'max_depth = %d,random_state = %d' % (4,0),
                             'RandomForestClassifier':'',
                             'svm.SVC':'kernel="rbf"', 
                             'KNeighborsClassifier':'n_neighbors = 4',
                             'LinearDiscriminantAnalysis':'solver = "lsqr",shrinkage = 0.02',
                             'QuadraticDiscriminantAnalysis':'',
                             'svm_linear.SVC':'kernel="linear"'
                            }
        
                model_lst = list(modelDict.keys())  
            
        
                #create lists for storing model scores for cross-validation
                models = []
                accuracy = []
                mpca = []
                f1_wt = []
        
                #create lists for storing model scores
                accuracy2 = []
                mpca2 = []
                f1_wt2 = []
                cost2 = []
                unique2 = []
                smote = []
            
            
            
                for key in modelDict.keys():
                    print(key)
                    models.append(key)
                    smote.append(key_1)
        
                    #define classifier with parameters including penalizing parameters
                    clf = eval('%s(%s)' % (key,modelDict[key]))
        
                    #define over and under sampling rates
                    over_sample_rate = int(0.7*arg3)
                    under_sample_rate = int(0.7*arg3)
                    #under_sample_rate = int(over_sample_rate + (over_sample_rate/2))
        
        
                    #assign sampling rates to dictionaries
                    under_sampling_dict = {}
                    over_sampling_dict = {1:over_sample_rate,2:over_sample_rate}
        
                    # define pipeline
                    over = eval('%s(%s)' % (key_1,SMOTE_dict[key_1]))
                    under = RandomUnderSampler(sampling_strategy=under_sampling_dict, random_state=1)
                    steps = [('o', over),('u',under), ('model',clf)]
                    pipeline = Pipeline(steps=steps)
        
        
                    #evaluate pipeline
        
                    # define cross-validation method for model evaluation
                    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)       
        
                    
                    #calculate cross-validated mean per class accuracy (recall macroaverage from classification report)
                    results = cross_validate(pipeline, arg1, arg2, scoring=['recall_macro','accuracy','f1_macro','f1_weighted'], cv=cv, n_jobs=-1,error_score='raise')
        
                    #append cross-validation results to respective lists
                    accuracy.append(results['test_accuracy'].mean())
                    mpca.append(results['test_recall_macro'].mean())
                    f1_wt.append(results['test_f1_weighted'].mean())
                    
        
        
        
        
                    #train each model with training data and predict on test data
        
                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = tts(arg1, arg2, test_size=0.03, random_state=arg4,stratify=arg2)
        
                    # Create a dictionary to store the indices of each label in the training set
                    label_indices = {}
                    for i in range(len(y_train)):
                        label = y_train[i]
                        if label not in label_indices:
                            label_indices[label] = []
                        label_indices[label].append(i)
        
                    # Create a list to store the 5 samples from each label
                    samples = []
                    for label in label_indices:
                        label_samples = label_indices[label][:5] # Get the first 5 indices for each label
                        for idx in label_samples:
                            samples.append(idx)
        
                    # Get the data and labels for the selected samples
                    X_test = X_train[samples]
                    y_test = y_train[samples]
                    
        
                    
                    print(np.unique(y_train))
                    #train models
                    model = pipeline.fit(X_train, y_train)  
                    
        
                    #predict on test data
                    y_hat = model.predict(X_test)      
                    print(classification_report(y_test, y_hat))
        
                    #print confusion matrix
                    CM = confusion_matrix(y_test,y_hat)
                    print(CM)
        
        
                    #capture classification accuracy metrics
                    report_dict = classification_report(y_test,y_hat,output_dict=True)
        
                    #mean per class accuracy
                    mpca2.append(report_dict['macro avg']['recall']) #mean per class accuracy
        
                    #return f1 score
                    f1_wt2.append(report_dict['weighted avg']['f1-score'])
        
                    #accuracy
                    accuracy2.append(report_dict['accuracy'])   
        
                    #misclassification cost
                    cost2.append(misclassification_cost(y_test,y_hat))
        
                    #unique label predictions
                    unique2.append(np.unique(y_hat))
           
                
                
                #create and populate dataframe with cross-validation results
                df_scores = pd.DataFrame()
                df_scores['Model'] = models
                df_scores['CV Accuracy'] = accuracy
                df_scores['CV MPCA'] = mpca
                df_scores['CV F1_weighted'] = f1_wt
                
        
        
                #create and populate dataframe with trained model results
                df_scores2 = pd.DataFrame()
                df_scores2['SMOTE MDL'] = smote
                df_scores2['Model'] = models
                df_scores2['Accuracy'] = accuracy2
                df_scores2['MPCA'] = mpca2
                df_scores2['F1_weighted'] = f1_wt2
                df_scores2['Misclassification_Cost'] = cost2
                df_scores2['Unique Predictions'] = unique2
                
                mb_output = mb_output.append(df_scores2)
                
                
                print(key_1)
                
            
            return(df_scores2,model,X_test,y_test,X_train,y_train,df_scores,mb_output)
        
        
        # M0 Model
        
        def smote_classifier_m0(arg1, arg2, arg3, arg4):
    
        '''
        arg1: Input array (features) - Training data (features)
        arg2: Target array  - Training data (labels)
        arg3: number of items in the majority class (max count)
        arg4: random seed for splitting data
    
        '''
        #create dictionary to store models and default parameters
        
        
        SMOTE_dict = {'SMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),
                      'BorderlineSMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),        
                      #'ADASYN': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),
                     'SVMSMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42)}
        
        
        
        #create empty dataframe for storing classification results
        m0_output = pd.DataFrame()
        
        for key_1 in SMOTE_dict.keys():
            
            print(key_1)
        
    
        
        
            modelDict = {'DecisionTreeClassifier':'max_depth = %d,random_state = %d' % (4,0),
                         'RandomForestClassifier':'',
                         'svm.SVC':'kernel="rbf"', 
                         'KNeighborsClassifier':'n_neighbors = 4',
                         'LinearDiscriminantAnalysis':'solver = "lsqr",shrinkage = 0.02',
                         'QuadraticDiscriminantAnalysis':'',
                         'svm_linear.SVC':'kernel="linear"'
                        }
    
            model_lst = list(modelDict.keys())  
    
    
    
    
            #create lists for storing model scores
            #list names with 2 at the end are not cross-validated scores
            #list names without 2 are cross-validated
            models = []
            accuracy = []
            accuracy2 = []
            mpca = []  
            mpca2 = []
            f1_wt2 = []
            cost2 = []
            f1_wt = []
            unique_predictions = []
            smote = []
        
            for key in modelDict.keys():
                print(key)
                models.append(key)
                smote.append(key_1)
    
                #define classifier with parameters including penalizing parameters
                clf = eval('%s(%s)' % (key,modelDict[key]))
    
                #define over and under sampling rates
                over_sample_rate = int(0.7*arg3)
                under_sample_rate = int(0.7*arg3)
                #under_sample_rate = int(over_sample_rate + (over_sample_rate/2))
    
    
    
                #assign sampling rates to dictionaries
                under_sampling_dict = {}
                over_sampling_dict = {1:over_sample_rate}
    
                # define pipeline
                over = eval('%s(%s)' % (key_1,SMOTE_dict[key_1]))
                under = RandomUnderSampler(sampling_strategy=under_sampling_dict,random_state=1)
                steps = [('o', over), ('u', under), ('model',clf)]
                pipeline = Pipeline(steps=steps)
    
    
                #evaluate pipeline
    
                # define cross-validation method for model evaluation
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)       
    
                
                #calculate cross-validated mean per class accuracy (recall macroaverage from classification report)
                results = cross_validate(pipeline, arg1, arg2, scoring=['recall_macro','accuracy','f1_macro','f1_weighted'], cv=cv, n_jobs=-1,error_score='raise')
    
                #append cross-validation results to respective lists
                accuracy.append(results['test_accuracy'].mean())
                mpca.append(results['test_recall_macro'].mean())
                f1_wt.append(results['test_f1_weighted'].mean())
                
    
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = tts(arg1, arg2, test_size=0.03, random_state=arg4,stratify=arg2)
    
                # Create a dictionary to store the indices of each label in the training set
                label_indices = {}
                for i in range(len(y_train)):
                    label = y_train[i]
                    if label not in label_indices:
                        label_indices[label] = []
                    label_indices[label].append(i)
    
                # Create a list to store the 5 samples from each label
                samples = []
                for label in label_indices:
                    label_samples = label_indices[label][:5] # Get the first 5 indices for each label
                    for idx in label_samples:
                        samples.append(idx)
    
                # Get the data and labels for the selected samples
                X_test = X_train[samples]
                y_test = y_train[samples]
    
    
                #train each model with training data and predict on test data
                model = pipeline.fit(X_train, y_train)       
                y_hat = model.predict(X_test)
                print(classification_report(y_test, y_hat))
    
    
                #print classification report
                report_dict = classification_report(y_test,y_hat,output_dict=True)
                print(classification_report(y_test,y_hat))
    
                #mean per-class accuracy
                mpca2.append(report_dict['macro avg']['recall']) #mean per class accuracy
    
                #return f1 score
                f1_wt2.append(report_dict['weighted avg']['f1-score'])
                #f1_macro.append(report_dict['macro avg']['f1-score'])
    
    
                #accuracy
                print('%s Accuracy: %s' % (key,accuracy_score(y_hat,y_test)))        
                accuracy2.append(accuracy_score(y_hat,y_test))
    
    
    
                #uniqueness of the predictions
                unique_predictions.append(np.unique(y_hat))
                print(np.unique(y_hat))
    
                #print confusion matrix
                CM = confusion_matrix(y_test,y_hat)
                print(CM)
    
                #misclassification cost
                cost2.append(misclassification_cost(y_test,y_hat))
    
    
            
            df_scores = pd.DataFrame()
            df_scores['Model'] = models
            df_scores['CV Accuracy'] = accuracy
            df_scores['CV MPCA'] = mpca
            df_scores['CV F1_weighted'] = f1_wt
            
            
            df_scores2 = pd.DataFrame()
            df_scores2['SMOTE MDL'] = smote
            df_scores2['Model'] = models
            df_scores2['Accuracy'] = accuracy2
            df_scores2['MPCA'] = mpca2
            df_scores2['F1_weighted'] = f1_wt2
            df_scores2['Misclassification_Cost'] = cost2
            df_scores2['unique predictions'] = unique_predictions
            
            print(key_1)
            
            m0_output = m0_output.append(df_scores2)
         
        
        return(df_scores2,model,X_test,y_test,X_train,y_train,df_scores,m0_output)
    
    # M1 Model
    
    def smote_classifier_m1(arg1, arg2, arg3, arg4):
    
        '''
        arg1: Input array (features) - Training data (features)
        arg2: Target array  - Training data (labels)
        arg3: number of items in the majority class (max count)
        arg4: random seed for splitting data
    
        '''
        
        
        #create empty datafram for storing classification results
        m1_output = pd.DataFrame()
        
        #create dictionary to store models and default parameters
        SMOTE_dict = {'SMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),
                      'BorderlineSMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),        
                      #'ADASYN': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),
                     'SVMSMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42)}
        
        
        for key_1 in SMOTE_dict.keys():
            
            print(key_1)
    
        
        
            modelDict = {'DecisionTreeClassifier':'class_weight="balanced",max_depth = %d,random_state = %d' % (4,0),
                         'RandomForestClassifier':'class_weight="balanced"',
                         'svm.SVC':'kernel="rbf"', 
                         'KNeighborsClassifier':'n_neighbors = 4',
                         'LinearDiscriminantAnalysis':'solver = "lsqr",shrinkage = 0.02',
                         'QuadraticDiscriminantAnalysis':'',
                         'svm_linear.SVC':'class_weight = "balanced"'
                        }
    
            model_lst = list(modelDict.keys()) 
    
    
    
    
            #create lists for storing model scores
            models = []
            accuracy = []
            mpca = []
            f1_wt = []
    
            accuracy2 = []
            mpca2 = []
            f1_wt2 = []
            cost2 = [] 
            unique_predictions = []
            smote = []
        
            for key in modelDict.keys():
                print(key)
                models.append(key)
                smote.append(key_1)
    
                #define classifier with parameters including penalizing parameters
                clf = eval('%s(%s)' % (key,modelDict[key]))
    
                #define over and under sampling rates
                over_sample_rate = int(0.85*arg3)
                under_sample_rate = int(0.7*arg3)
                #under_sample_rate = int(over_sample_rate + (over_sample_rate/2))
    
    
                #assign sampling rates to dictionaries
                under_sampling_dict = {}
                over_sampling_dict = {1:over_sample_rate}
    
                # define pipeline
                over = eval('%s(%s)' % (key_1,SMOTE_dict[key_1]))
                under = RandomUnderSampler(sampling_strategy=under_sampling_dict,random_state=1)
                steps = [('o', over), ('u', under), ('model',clf)]
                pipeline = Pipeline(steps=steps)
    
    
                #evaluate pipeline
    
                # define cross-validation method for model evaluation
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)       
    
                
                #calculate cross-validated mean per class accuracy (recall macroaverage from classification report)
                results = cross_validate(pipeline, arg1, arg2, scoring=['recall_macro','accuracy','f1_macro','f1_weighted'], cv=cv, n_jobs=-1,error_score='raise')
    
                #print(results)
    
                #append cross-validation results to respective lists
                accuracy.append(results['test_accuracy'].mean())
                mpca.append(results['test_recall_macro'].mean())
                f1_wt.append(results['test_f1_weighted'].mean())
                
    
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = tts(arg1, arg2, test_size=0.03, random_state=arg4,stratify=arg2)
    
                # Create a dictionary to store the indices of each label in the training set
                label_indices = {}
                for i in range(len(y_train)):
                    label = y_train[i]
                    if label not in label_indices:
                        label_indices[label] = []
                    label_indices[label].append(i)
    
                # Create a list to store the 5 samples from each label
                samples = []
                for label in label_indices:
                    label_samples = label_indices[label][:5] # Get the first 5 indices for each label
                    for idx in label_samples:
                        samples.append(idx)
    
                # Get the data and labels for the selected samples
                X_test = X_train[samples]
                y_test = y_train[samples]
    
                #train each model with training data and predict on test data
                model = pipeline.fit(X_train, y_train)       
                y_hat = model.predict(X_test)
    
                #print classification report
                report_dict = classification_report(y_test,y_hat,output_dict=True)
                print(classification_report(y_test,y_hat))
    
                #mean per-class accuracy
                mpca2.append(report_dict['macro avg']['recall']) #mean per class accuracy
    
                #return f1 score
                f1_wt2.append(report_dict['weighted avg']['f1-score'])
    
                #accuracy
                print('%s Accuracy: %s' % (key,accuracy_score(y_hat,y_test)))        
                accuracy2.append(accuracy_score(y_hat,y_test))
    
                #print confusion matrix
                CM = confusion_matrix(y_test,y_hat)
                print(CM)      
    
                #uniqueness of the predictions
                unique_predictions.append(np.unique(y_hat))
                print(np.unique(y_hat))
    
                #misclassification cost
                cost2.append(misclassification_cost(y_test,y_hat))
    
    
    
            #show distribution of training and test labels to assess balance of data
            labels, counts = np.unique(y_train, return_counts=True)
            plt.bar(labels,counts,align='center')
            plt.xlabel('Class')
            plt.ylabel('Distribution of Training Class')
            #plt.xticks([0,1,2])
            plt.show()
            plt.close()
    
            labels, counts = np.unique(y_test, return_counts=True)
            plt.bar(labels,counts,align='center')
            plt.xlabel('Class')
            plt.ylabel('Distribution of Test Class')
            #plt.xticks([0,1,2])
            plt.show()
            plt.close()
    
    
            df_scores2 = pd.DataFrame()
            df_scores2['SMOTE MDL'] = smote
            df_scores2['Model'] = models
            df_scores2['Accuracy'] = accuracy2
            df_scores2['MPCA'] = mpca2
            df_scores2['F1_weighted'] = f1_wt2
            df_scores2['Misclassification_Cost'] = cost2
            df_scores2['unique predictions'] = unique_predictions
    
    
            
            df_scores = pd.DataFrame()
            df_scores['Model'] = models
            df_scores['CV Accuracy'] = accuracy
            df_scores['CV MPCA'] = mpca
            df_scores['CV F1_weighted'] = f1_wt
            
            
            
            print(key_1)
            
            #append results to dataframe
            m1_output = m1_output.append(df_scores2)
    
        
        return(df_scores2,model,X_test,y_test,X_train,y_train,df_scores,m1_output)
    
    # M2 Model
    
    def smote_classifier_m2(arg1, arg2, arg3, arg4):
    
        '''
        arg1: Input array (features) - Training data (features)
        arg2: Target array  - Training data (labels)
        arg3: number of items in the majority class (max count)
        arg4: random seed for splitting data
    
        '''
        
        #crate empty dataframe for classification results
        m2_output = pd.DataFrame()
        
        #create dictionary to store models and default parameters
        SMOTE_dict = {'SMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),
                      'BorderlineSMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),        
                      #'ADASYN': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42),
                     'SVMSMOTE': 'sampling_strategy = over_sampling_dict,random_state = %d' % (42)}
        
        
        for key_1 in SMOTE_dict.keys():
            
            print(key_1)
    
        
        
            modelDict = {'DecisionTreeClassifier':'max_depth = %d,random_state = %d' % (4,0),
                         'RandomForestClassifier':'',
                         'svm.SVC':'kernel="rbf"' ,
                         'KNeighborsClassifier':'n_neighbors = 4',
                         'LinearDiscriminantAnalysis':'solver = "lsqr",shrinkage = 0.02',
                         'QuadraticDiscriminantAnalysis':'',
                         'svm_linear.SVC':'kernel = "linear"'
                        }
    
            model_lst = list(modelDict.keys())  
    
    
    
    
            #create lists for storing model scores
            models = []
            accuracy = []
            mpca = []
            f1_wt = []
    
            accuracy2 = []
            mpca2 = []
            f1_wt2 = []
            cost2 = [] 
            unique_predictions = []
            smote = []
        
            for key in modelDict.keys():
                print(key)
                models.append(key)
                smote.append(key_1)
    
                #define classifier with parameters including penalizing parameters
                clf = eval('%s(%s)' % (key,modelDict[key]))
    
    
    
                #define over and under sampling rates
                over_sample_rate = int(0.85*arg3)
                #under_sample_rate = int(0.5*arg3)
    
                #assign sampling rates to dictionaries
                #under_sampling_dict = {3:under_sample_rate}
                over_sampling_dict = {}
    
                # define pipeline
                over = eval('%s(%s)' % (key_1,SMOTE_dict[key_1]))
                #over = SMOTE(random_state = 1)
    
                #under = RandomUnderSampler(sampling_strategy=under_sampling_dict)
                under = RandomUnderSampler(random_state=1)
    
                steps = [('o', over), ('u', under), ('model',clf)]
                pipeline = Pipeline(steps=steps)
    
    
                #evaluate pipeline
    
    
                # define cross-validation method for model evaluation
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)       
    
                
                
                #calculate cross-validated mean per class accuracy (recall macroaverage from classification report)
                results = cross_validate(pipeline, arg1, arg2, scoring=['recall_macro','accuracy','f1_macro','f1_weighted'], cv=cv, n_jobs=-1,error_score='raise')
    
                #append cross-validation results to respective lists
                accuracy.append(results['test_accuracy'].mean())
                mpca.append(results['test_recall_macro'].mean())
                f1_wt.append(results['test_f1_weighted'].mean())
                
    
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = tts(arg1, arg2, test_size=0.03, random_state=arg4,stratify=arg2)
    
                # Create a dictionary to store the indices of each label in the training set
                label_indices = {}
                for i in range(len(y_train)):
                    label = y_train[i]
                    if label not in label_indices:
                        label_indices[label] = []
                    label_indices[label].append(i)
    
                # Create a list to store the 5 samples from each label
                samples = []
                for label in label_indices:
                    label_samples = label_indices[label][:5] # Get the first 5 indices for each label
                    for idx in label_samples:
                        samples.append(idx)
    
                # Get the data and labels for the selected samples
                X_test = X_train[samples]
                y_test = y_train[samples]
                
    
                #train each model with training data and predict on test data
                model = pipeline.fit(X_train, y_train)
                y_hat = model.predict(X_test)
                print(classification_report(y_test, y_hat))
    
                #print classification report
                report_dict = classification_report(y_test,y_hat,output_dict=True)
                print(classification_report(y_test,y_hat))
    
                #mean per-class accuracy
                mpca2.append(report_dict['macro avg']['recall']) #mean per class accuracy
    
                #return f1 score
                f1_wt2.append(report_dict['weighted avg']['f1-score'])
    
                #accuracy
                print('%s Accuracy: %s' % (key,accuracy_score(y_hat,y_test)))        
                accuracy2.append(accuracy_score(y_hat,y_test))
    
                #print confusion matrix
                CM = confusion_matrix(y_test,y_hat)
                print(CM)      
    
                #uniqueness of the predictions
                unique_predictions.append(np.unique(y_hat))
                print(np.unique(y_hat))
    
                #misclassification cost
                cost2.append(misclassification_cost(y_test,y_hat))
            
            
            
            df_scores2 = pd.DataFrame()
            df_scores2['SMOTE MDL'] = smote
            df_scores2['Model'] = models
            df_scores2['Accuracy'] = accuracy2
            df_scores2['MPCA'] = mpca2
            df_scores2['F1_weighted'] = f1_wt2
            df_scores2['Misclassification_Cost'] = cost2
            df_scores2['unique predictions'] = unique_predictions   
    
            
            
            
            df_scores = pd.DataFrame()
            df_scores['Model'] = models
            df_scores['CV Accuracy'] = accuracy
            df_scores['CV MPCA'] = mpca
            df_scores['CV F1_weighted'] = f1_wt
            
            print(key_1)
    
            m2_output = m2_output.append(df_scores2)
        
        return(df_scores2,model,X_test,y_test,X_train,y_train,df_scores,m2_output)
    
    #Train Models
    
    os.chdir(args.output_path)
    
    # Base Model
    
    mb_results = pd.DataFrame()

    rdm_seed_lst = [42,43,44,45,46]
    
    for i in range(len(rdm_seed_lst)): 
        mb = smote_classifier_base(X,y,max_count_base,rdm_seed_lst[i])
        mb_results = mb_results.append(mb[7])
        
    mb_results.to_excel('mb_results.xlsx')
    
    # M0 Model
    
    m0_results = pd.DataFrame()

    rdm_seed_lst = [42,43,44,45,46]
    
    for i in range(len(rdm_seed_lst)):
        
        
        m0 = smote_classifier_m0(X_m0,y_m0,max_count_m0,rdm_seed_lst[i])
        m0_results = m0_results.append(m0[7])
        
    m0_results.to_excel('m0_results.xlsx')
    
    # M1 Model
    
    m1_results = pd.DataFrame()

    rdm_seed_lst = [42,43,44,45,46]
    
    for i in range(len(rdm_seed_lst)):
            
        m1 = smote_classifier_m1(X_m1,y_m1,max_count_m1,rdm_seed_lst[i])
        m1_results = m1_results.append(m1[7])
        
    m1_results.to_excel('m1_results.xlsx')
    
    
    # M2 Model
    
    m2_results = pd.DataFrame()

    rdm_seed_lst = [42,43,44,45,46]
    
    for i in range(len(rdm_seed_lst)):
           
        m2 = smote_classifier_m2(X_m2,y_m2,max_count_m2,rdm_seed_lst[i])
        m2_results = m2_results.append(m2[7])
        
    m2_results.to_excel('m2_results.xlsx')
    
    
    # Hierarchical Model
    
    #hierarchical classification of validation data

    rdm_seed_lst = [42,43,44,45,46]
    
    
    seed = []
    mpca = []
    f1_wt = []
    cost = []
    accuracy = []
    unique_predictions = []
    
    
    
    for i in range(len(rdm_seed_lst)):
        seed.append(rdm_seed_lst[i])
        
        #mb = smote_classifier_base(X,y,max_count_base,rdm_seed_lst[i
        X_train, X_test, y_train, y_test = tts(X,y, random_state=i, stratify = y)
        mbx = [X_train,  y_train, X_test, y_test]
    
        pred_list = []
        for i in range(0,len(mbx[2])):
            if m0[1].predict(mbx[2])[i] == 1:    #idc score is 1 or 2
                pred = m1[1].predict(mbx[2])[i]  #use model 1 to classify as 1 or 2
                pred_list.append(pred)                          
            elif m0[1].predict(mbx[2])[i] == 2:
                pred = 3
                pred_list.append(pred)
            elif m0[1].predict(mbx[2])[i] == 3:   #idc score is 4 or 5
                pred = m2[1].predict(mbx[2])[i]   #use model 2 to classify as 4 or 5
                pred_list.append(pred)
        pred_list = np.array(pred_list)                         
    
        print(classification_report(mbx[3],pred_list))
        
        #print classification report
        report_dict = classification_report(mbx[3],pred_list,output_dict=True)
    
        #mean per-class accuracy
        mpca.append(report_dict['macro avg']['recall']) #mean per class accuracy
    
        #return f1 score
        f1_wt.append(report_dict['weighted avg']['f1-score'])
    
        #accuracy
        accuracy.append(accuracy_score(pred_list,mbx[3]))       
    
        #uniqueness of the predictions
        unique_predictions.append(np.unique(pred_list))
    
        #misclassification cost
        cost.append(misclassification_cost(mbx[3],pred_list))
        
    df_hierarchy_cv = pd.DataFrame()
    df_hierarchy_cv['Seed'] = seed
    df_hierarchy_cv['Accuracy'] = accuracy
    df_hierarchy_cv['MPCA'] = mpca
    df_hierarchy_cv['F1_weighted'] = f1_wt
    df_hierarchy_cv['Misclassification Cost'] = cost
    
    df_hierarchy_cv.to_excel('hierarchy_results.xlsx')



    
 
    
            
    
    
