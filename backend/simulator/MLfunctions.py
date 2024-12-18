###############################################################################################
# Developed by: Sebastian Peñaherrera
# Date: 07/07/2021
# Description: This file contains a set of functions that are used to train and assess
# machine learning models. The functions are used to train models, assess their performance
# and save the models for future usage. The functions are also used to load the models and
# evaluate their performance. The functions are also used to save and load the models and
# their performance metrics. 

# ---------------------------------------------------------------------------------------------

# Import the deprecated decorator
import warnings
from warnings import warn
warnings.simplefilter("ignore")

# General purpose packages
import pandas as pd
import numpy as np
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import MLfunctions as fun
import math
from timeit import default_timer as timer
import random
import functools
from tqdm import tqdm
import joblib

# Import packages from Sci-kit Learn
#METRICS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.metrics import normalized_mutual_info_score

#UTILITIES
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

#FEATURE ENGINEERING
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, mutual_info_regression 
from sklearn.feature_selection import SelectKBest, f_regression

#REGRESSION
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

#CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Import VIF from statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from termcolor import colored

# Import json package
import json

class Functions:
    
    @staticmethod
    def train_models(X_train_FS: pd.DataFrame, X_train_FE: pd.DataFrame, X_test_FS: pd.DataFrame, X_test_FE: pd.DataFrame, y_train: pd.DataFrame , y_test: pd.DataFrame, 
                     targets: list, algorithms: list, feature_modes: list, feature_selection: str = 'kbest', gridsearch_score_method: str = 'multiple', save_model: bool = False, 
                     use_kqi: bool = False, KQI_type: dict = {}):
        
        '''
        This method trains the models for the different targets, algorithms and feature engineering modes.
        This function iterates over the targets, feature engineering modes and algorithms to train the models and store the stats in a dictionary. This can be used to automate the training of the models.
        Some of the configurations are hardcoded in the function, such as the gridsearch_score_method, feature_selection, save_model and use_kqi. These configurations can be modified in the function.
        Some other parameters are written in the configuration file. If the dataset changes, the configuration file must be updated.

        Inputs:
        X_train_FS: DataFrame with the features of the training set with feature selection
        X_train_FE: DataFrame with the features of the training set with feature extraction
        X_test_FS: DataFrame with the features of the testing set with feature selection
        X_test_FE: DataFrame with the features of the testing set with feature extraction
        y_train: DataFrame with the target of the training set
        y_test: DataFrame with the target of the testing set
        targets: list, list with the target variables
        algorithms: list, list with the algorithms
        feature_modes: list, list with the feature engineering modes
        feature_selection: str, feature selection mode
        gridsearch_score_method: str, scoring method for the GridSearchCV
        save_model: bool, if True, save the model
        use_kqi: bool, if True, use the KQI in the model
        KQI_type: dict, dictionary with the type of KQI

        Outputs:
        stats_target: dict, dictionary with the stats of the models
        '''

        # Create a dictionary to store the stats by target
        stats_target = {}

        # Iterate over the configurations
        for target in tqdm(targets, desc='Targets'):
            # Print the target
            print(colored(f'Target: {target}', 'black', 'on_yellow', attrs=['bold']))
            
            # Create a dictionary to store the stats
            stats_feature = {}

            # Iterate over the feature modes
            for feature_mode in tqdm(feature_modes, desc='Feature engineering mode'):
                print(colored(f'Feature mode: {feature_mode}', 'black', 'on_blue', attrs=['bold']))
                # Initialize the dictionary to store the stats for the algorithm
                stats_algorithm = {}

                # Select the input dataset according to the feature mode
                if feature_mode == 'Feature_selection':
                    X_train = X_train_FS
                    X_test = X_test_FS
                else:
                    X_train = X_train_FE
                    X_test = X_test_FE

                # Iterate over the algorithms
                for algorithm in tqdm(algorithms, desc='Algorithms', leave=False):
                    print(colored(f'Algorithm: {algorithm}', 'black', 'on_green', attrs=['bold']))

                    # Get the type of ml_mode depending on the target type. If the target is continuous, the ml_mode is regression. If the target is categorical, the ml_mode is classification
                    ml_mode = Functions.map_ml_mode(target, KQI_type)

                    # Train the model
                    stats, model, grid = Functions.train_single_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, algorithm=algorithm, feature_mode=feature_mode, ml_mode=ml_mode, 
                                                        target=target, feature_selection=feature_selection, gridsearch_score_method=gridsearch_score_method, save_model=save_model, use_kqi=use_kqi)

                    # If stats is not empty, store the stats in the dictionary
                    if stats:
                        stats_algorithm[algorithm] = stats

                # Store the stats in a dictionary for the feature mode
                feature_mode = Functions.map_feature_eng_mode(feature_mode)

                # If the stats_algorithm is not empty, store the stats in the dictionary
                if stats_algorithm:
                    stats_feature[feature_mode] = stats_algorithm

            # Store the stats in a dictionary for the target if not empty
            if stats_feature:
                stats_target[target] = stats_feature

                # Save the individual stats in a json file
                Functions.save_stats({target: stats_feature}, target)

        # Save the overall stats in a json file if not empty
        if stats_target:
            Functions.save_stats(stats_target, 'overall')

        # Return the stats
        return stats_target, model
    
#FUNCIONES PARA  "train_models"
    @staticmethod
    def map_ml_mode(target: str, KQI_type: dict):
        '''
        This method maps the machine learning mode depending on the target type. If the target is continuous, the ml_mode is regression. If the target is categorical, the ml_mode is classification.

        Inputs:
        target: str, target variable
        KQI_type: dict, dictionary with the type of KQI

        Outputs:
        ml_mode: str, machine learning mode
        '''

        # Get the type of ml_mode depending on the target type. If the target is continuous, the ml_mode is regression. If the target is categorical, the ml_mode is classification
        if KQI_type[target] == 'continuous':
            ml_mode = 'regression'
        else:
            ml_mode = 'classification'

        # Return the ml_mode
        return ml_mode

    @staticmethod
    def map_feature_eng_mode(feat_eng_mode: str|list):
        '''
        This methods maps the feature engineering mode to a short version. The short version is used to name the models and the files.

        Inputs:
        feat_eng_mode: str, feature engineering mode to map

        Returns:
        str, short version of the feature engineering mode
        '''

        # If feat_eng_mode is 'FS1', return 'FS'
        if feat_eng_mode == 'Feature_selection':
            return 'FS'
        # If feat_eng_mode is 'PCA', return 'PCA'
        elif feat_eng_mode == 'Feature_extraction':
            return 'FE'
        # If feat_eng_mode is 'No_FE', return 'No_FE'
        elif feat_eng_mode == 'No_FE':
            return 'No_FE'
        # If string is empty, return 'No_FE'
        else:
            return 'No_FE'
    
    @staticmethod
    def save_stats(stats: dict, target: str, path: str = './stats_models', message: bool = False):
        '''
        This method saves the statistics of the model in a json file.

        Inputs:
        target: str, name of the target
        path: str, path to the directory where the stats will be saved

        Returns:
        None
        '''

        # Import the necessary packages
        import json

        # Check if the data path exists
        if Functions.CheckDirectory(path, msg=message):
            # Create a new directory for the split datasets with the today's date inside the data directory
            today = pd.to_datetime('today').strftime('%Y-%m-%d')
            Functions.CheckDirectory(f'{path}/{today}', msg=message)

        # Save the dictionary in a json file
        fileName = f'{path}/{today}/stats_{target}'
        Functions.SaveJson(stats, fileName)
        
    @staticmethod
    def train_single_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, algorithm: str, target: str, feature_mode: str, ml_mode: str, 
                           feature_selection :str = 'kbest', gridsearch_score_method = 'multiple', save_model: bool = False, use_kqi: bool = False):
        '''
        This method trains a single model with the best hyperparameters found through the GridSearchCV.

        Inputs:
        X_train: DataFrame with the features of the training set
        y_train: DataFrame with the target of the training set
        X_test: DataFrame with the features of the testing set
        y_test: DataFrame with the target of the testing set
        algorithm: str, algorithm to create the model
        target: str, target variable
        feature_mode: str, feature engineering mode
        feature_selection: str, feature selection mode
        ml_mode: str, machine learning mode 
        gridsearch_score_method: str, scoring method for the GridSearchCV
        save_model: bool, if True, save the model
        use_kqi: bool, if True, use the KQI in the model

        Outputs:
        stats: dict, dictionary with the stats of the model
        best_model: model, best ML model with hyperparamterer tuning
        '''
        
        # Initialize the start time
        start = timer()

        # Create a model and hyperparameters grid
        model, param_grid = Functions.CreateModelAndHyperparamsGrid(X_train=X_train, algorithm=algorithm, feat_eng_mode=feature_mode, ml_type=ml_mode)

        # Check if the model is different from None
        if model is not None:
            # CreatePipeline
            pipe = Functions.CreatePipeline(X_train, model, param_grid, feature_mode, feat_sel_mode=feature_selection, ml_mode=ml_mode)

            # Train the model through the GridSearchCV
            print(f'Training the model for {target}...')
            grid, best_model, best_params = Functions.TrainGridSearchCV(X_train, y_train[target], pipe, param_grid, gridsearch_score_method=gridsearch_score_method)
            print(f'Best model: \n\t{best_params}')

            # Determine the training time
            end = timer()
            tTime = end - start
            print(f'Training time: {tTime}')

            # Fetch the importance of the features
            if feature_mode == 'Feature_selection':
                print(f'Calculating the feature importance for this model...')
                feature_importance = Functions.GetFeatureImportanceInfo(grid)
            else:
                feature_importance = None

            # Fetch the grid search results
            cv_results = Functions.get_cv_results(grid)

            # Transform the unsupported data types to python normal data types
            cv_results = json.loads(cv_results.to_json(orient='columns')) # The DataFrame is serialized to json and then deserialized to a dictionary
            best_params = pd.Series(best_params).to_dict() # The object is transformed to a pandas Series and then to a dictionary
  
            # Evaluate the final model
            model_evaluation = Functions.AssessModel(best_model, X_test, y_test[target])
            print(f'Model assessment: \n\t{model_evaluation}')

            # Create a dictionary with the results
            stats = Functions.get_model_stats(algorithm, target, ml_mode, feature_importance, cv_results, model_evaluation, best_params, tTime)
            
            # Save the model
            if save_model:
                Functions.save_ml_model(model_obj=best_model, name=algorithm, target=target, ml_mode=ml_mode, feat_eng_mode=feature_mode, use_kqi=use_kqi)

            # Return the stats and the best model
            return stats, best_model, grid
        else:
            print(f'ERROR: The model {algorithm} is not recognized. Passing to the next model...')
            end = timer()
            return {}, None, {}
        
    
#FUNCIONES PARA "train_single_model" 

    @staticmethod
    def CreateModelAndHyperparamsGrid(X_train: pd.DataFrame = pd.DataFrame(), algorithm: str = '', feat_eng_mode: str = '', ml_type: str = 'regression'):
        '''
        This method creates a parameter grid for GridSearchCV tuning in function of the algorithm.

        Inputs:
        X_train: DataFrame with the features of the training set
        algorithm: str, algorithm to create the parameter grid
        feat_eng_mode: str, feature engineering mode to create the parameter grid
        ml_type: str, type of machine learning model (regression or classification)

        Returns:
        model: model, model object
        '''

        # Get the number of features to be used for prediction
        n_feat = X_train.shape[1]

        # Select the mode of machine learning type
        if ml_type == 'regression':

            # Create a model object with an parameter grid for GridSearchCV tuning
            if algorithm == 'RF':
                    # Create a model object
                    model = RandomForestRegressor(random_state = 0)
                    # Define the parameter space that will be searched over
                    param_grid = {
                                    'model__n_estimators' : np.arange(10, 101, 10),
                                    'model__max_depth' : np.arange(5, 10)
                                }

            elif algorithm == 'RR':
                # Create a model object
                model = Ridge(random_state = 0)
                # Define the parameter space that will be searched over
                param_grid = {
                                'model__alpha' : np.logspace(-5, 5, 10),
                                'model__fit_intercept' : [True, False]
                            }

            elif algorithm == 'SVR':
                # Create a model object
                model = SVR(max_iter = 5000)
                # define the parameter space that will be searched over
                param_grid = {
                                'model__kernel' : ['poly','rbf','sigmoid'],
                                'model__degree' : np.arange(1,7,1),
                                'model__epsilon' : np.array([0.01, 0.1, 0.5, 1]),
                                'model__C' : np.array([0.1, 1, 10, 100])
                                #'model__max_iter : 200
                            }

            elif algorithm == 'KNR':
                # Create a model object
                model = KNeighborsRegressor()
                # define the parameter space that will be searched over
                param_grid = {
                                'model__leaf_size' : np.arange(10,31,10),
                                'model__n_neighbors' : np.arange(2,7,2),
                                'model__p' : np.arange(1,3,1)
                            }

            elif algorithm == 'NN':
                # Create a model object
                model = MLPRegressor(max_iter = 5000, random_state = 0)
                # define the parameter space that will be searched over
                param_grid = {
                                'model__alpha' : np.array([0.0001, 0.0003, 0.001, 0.003, 0.01]),
                                #'model__hidden_layer_sizes' : [(100,),(100,100),(50,50,50)]
                                #'model__hidden_layer_sizes' : [(30,),(35,),(40,),(45,),(50,)]
                                #'model__hidden_layer_sizes' : [(30,),(40,),(50,),(60,),(70,),(80,),(90,)]
                                'model__hidden_layer_sizes' : [(80,),(100,),(80,80),(100,100),(80,80,80),(100,100,100),(200,200,200)] 
                                #model__hidden_layer_sizes' : [(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90)]
                            }

            elif algorithm == 'ABR':
                # Create a model object
                model = AdaBoostRegressor(random_state = 0)
                # define the parameter space that will be searched over
                param_grid = {
                                'model__n_estimators' : np.arange(50, 150, 25),
                                'model__learning_rate' : np.arange(0.0, 1.1, 0.333)
                            }
                
            else:
                # If the algorithm is not recognized, return None
                model = None
                param_grid = {}

            # Return the model object and the parameter grid
            return model, param_grid

        # If the type of algorithm is classification
        elif ml_type == 'classification':

            if algorithm == 'RF':
                # Create a model object
                model = RandomForestClassifier(random_state = 0)
                # Define the parameter space that will be searched over
                param_grid = {
                                'model__min_samples_leaf' : np.arange(10, 101, 10),
                                'model__max_depth' : np.arange(5, 10)
                            }


            elif algorithm == 'RR':
                # Create a model object
                model = RidgeClassifier(random_state = 0)
                # Define the parameter space that will be searched over
                param_grid = {
                                'model__alpha' : np.logspace(-5, 5, 10),
                                'model__fit_intercept' : [True, False]
                             }

            elif algorithm == 'SVC':
                # Create a model object
                model = SVC(gamma='scale', max_iter = 5000)
                # define the parameter space that will be searched over
                param_grid = {
                                'model__kernel' : ['poly','rbf','sigmoid'],
                                'model__degree' : np.arange(1,7,1),
                                'model__C' : np.array([0.1, 1, 10, 100])
                            }

            elif algorithm == 'KNR':
                # Create a model object
                model = KNeighborsClassifier()
                # define the parameter space that will be searched over
                param_grid = {
                                'model__leaf_size' : np.arange(20,41,5),
                                'model__n_neighbors' : np.arange(1,8,2),
                                'model__p' : np.arange(1,3,1)
                            }

            elif algorithm == 'NN':
                # Create a model object
                model = MLPClassifier(max_iter = 5000, random_state = 0)
                # define the parameter space that will be searched over
                param_grid = {
                                'model__alpha' : np.array([0.0001, 0.0003, 0.001, 0.003, 0.01]),
                                #'model__hidden_layer_sizes' : [(100,),(100,100),(50,50,50)]
                                #'model__hidden_layer_sizes' : [(30,),(35,),(40,),(45,),(50,)]
                                #'model__hidden_layer_sizes' : [(30,),(40,),(50,),(60,),(70,),(80,),(90,)]
                                'model__hidden_layer_sizes' : [(80,),(100,),(80,80),(100,100),(80,80,80),(100,100,100),(200,200,200)] 
                                #'model__hidden_layer_sizes' : [(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90)]
                            }

            elif algorithm == 'GNB':
                # Create a model object
                model = GaussianNB()
                # define the parameter space that will be searched over
                param_grid = {
                                'model__var_smoothing' : np.arange(0, 2e-9, 0.5e-9)
                            }

            else:
                # If the algorithm is not recognized, return None
                model = None
                param_grid = {}

            # Return the model object and the parameter grid
            return model, param_grid

        else:
            # If the type of machine learning model is not recognized, return None
            return None, {}

    @staticmethod
    def CreatePipeline(trainset, model, param_grid ={}, feat_eng_mode: str = '', feat_sel_mode: str = 'tree', ml_mode: str = 'regression'):
        '''
        This method creates a pipeline in function of the feature engineering mode.

        Inputs:
        trainset: DataFrame with the features of the training set
        model: model, model object
        param_grid: dict, dictionary with the parameters to search
        feat_eng_mode: str, feature engineering mode
        feat_sel_mode: str, feature selection mode (tree, kbest)
        ml_mode: str, machine learning mode (regression, classification)

        Returns:
        pipe: pipeline, pipeline object    
        '''
        # Get the number of features to be used for prediction
        n_feat = trainset.shape[1]

        #Create a pipeline. First dimension reduction with PCA, then Model training
        if feat_eng_mode == 'Feature_extraction':
            pipe = Pipeline([
                                ("reduce", PCA()),
                                ("model", model)
                            ])
            # Add the variables for the PCA to the parameter grid
            param_grid['reduce__n_components'] = np.arange(1, n_feat + 1, int(np.around(n_feat * 0.2))) # type: ignore
            
        elif feat_eng_mode == 'Feature_selection':

            # Create a pipeline with Tree-based feature selection
            if feat_sel_mode == 'tree':
                pipe = Pipeline([
                                    ('feature_selection', SelectFromModel(RandomForestRegressor(random_state = 10))),
                                    ('model', model)
                                ])
                # Add the variables for the Tree-based feature selection to the parameter grid
                param_grid['feature_selection__threshold'] = ['0.5*mean', 'mean', '1.5*mean'] # type: ignore
                
            # Create a pipeline with KBest feature selection
            else:
                # If the ml_mode is classification, use mutual_info_classif
                if ml_mode == 'classification':
                    pipe = Pipeline([
                                    ('feature_selection', SelectKBest(mutual_info_classif)),
                                    ('model', model)
                                ])
                    # Add the variables for the KBest feature selection to the parameter grid
                    param_grid['feature_selection__k'] = np.arange(1, n_feat + 1) # type: ignore

                # If the ml_mode is regression, use mutual_info_regression
                else:                  
                    pipe = Pipeline([
                                        ('feature_selection', SelectKBest(mutual_info_regression)),
                                        ('model', model)
                                    ])
                    # Add the variables for the KBest feature selection to the parameter grid
                    param_grid['feature_selection__k'] = np.arange(1, n_feat + 1) # type: ignore

        else:
            pipe = Pipeline([
                                ("model", model)
                            ])
            
        # Return the pipeline
        return pipe

    @staticmethod
    def TrainGridSearchCV(X_train, y_train, pipe, parameters_grid = {}, n_folds: int = 5, gridsearch_score_method: str|list = 'simple'):
        '''
        This method trains the pipeline with the GridSearchCV method. It looks for the best hyperparameters in the parameter grid.

        Inputs:
        X_train: DataFrame with the features of the training set
        y_train: DataFrame with the target of the training set
        pipe: pipeline, pipeline object
        parameters_grid: dict, dictionary with the parameters to search
        n_folds: int, number of folds for the cross-validation
        gridsearch_score_method: str, scoring method for the GridSearchCV

        Returns:
        grid: GridSearchCV, grid object
        best_model: dict, dictionary with the best hyperparameters
        '''

        # Infer the type of ml model used
        ml_type = Functions.get_type_ml_algorithm(pipe)

        # Check if gridsearch_score_method is a list
        if isinstance(gridsearch_score_method, list):
            refit = gridsearch_score_method[0]
            scoring = gridsearch_score_method

        # If is an instance of str, set refit to False
        else:
            # If it is a string, check if is set to simple or multiple scoring
            if gridsearch_score_method == 'simple':
                refit = True
                # If ml_type is regression, set the scoring method to 'neg_mean_absolute_error' as default
                if ml_type == 'regression':
                    scoring = make_scorer(Functions.mean_average_error_percentage, greater_is_better=False)
                # If ml_type is classification, set the scoring method to 'roc_auc' as default
                else:
                    scoring = 'roc_auc_ovr_weighted'

            elif gridsearch_score_method == 'multiple':
                # If ml_type is regression, set refit to 'neg_mean_absolute_error' as default metric for ranking
                if ml_type == 'regression':
                    refit = 'MAEP'
                    # Set the mutiple scoring methods
                    scoring = {'MAE': 'neg_mean_absolute_error', 
                               'MSE': 'neg_mean_squared_error', 
                               'MAPE': 'neg_mean_absolute_percentage_error', 
                               'R2': 'r2',
                               'EVS': 'explained_variance',
                               'RMSE': 'neg_root_mean_squared_error',
                               'MedAE': 'neg_median_absolute_error',
                               'MAEP' : make_scorer(Functions.mean_average_error_percentage, greater_is_better=False)
                               }
                # If ml_type is classification, set refit to 'roc_auc' as default metric for ranking
                else:
                    refit = 'F1_weighted'

                    # Check if the model has predict_proba method
                    if hasattr(pipe, 'predict_proba'):
                        # Set the mutiple scoring methods
                        scoring = {'AUC': 'roc_auc_ovr_weighted', 
                                    'Accuracy': 'accuracy', 
                                    'Precision': make_scorer(precision_score, average='weighted', zero_division=0, greater_is_better=True), 
                                    'Recall': 'recall_weighted',
                                    'F1_micro': 'f1_micro',
                                    'F1_macro': 'f1_macro',
                                    'F1_weighted': 'f1_weighted',
                                    'Log_Loss': 'neg_log_loss',
                                    'MCC': 'matthews_corrcoef',
                                    }
                    else:
                        scoring = { 
                                    'Accuracy': 'accuracy', 
                                    'Precision': make_scorer(precision_score, average='weighted', zero_division=0, greater_is_better=True), 
                                    'Recall': 'recall_weighted',
                                    'F1_micro': 'f1_micro',
                                    'F1_macro': 'f1_macro',
                                    'F1_weighted': 'f1_weighted',
                                    'MCC': 'matthews_corrcoef',
                                    }
            else:
                refit = False


        #Train the pipeline using GridSearchCV
        grid = GridSearchCV(pipe, param_grid = parameters_grid, cv=n_folds, n_jobs=-1, scoring=scoring, refit=refit)
        grid.fit(X_train, y_train)

         #Get the best hyperparameters
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        # Return the grid
        return grid, best_model, best_params
    
    @staticmethod
    def GetFeatureImportanceInfo(grid):
        '''
        This method returns the feature importance of the best model from the GridSearchCV with the stats of the models trained with the best parameters but with different number of features. 

        Inputs:
        grid: GridSearchCV, grid object

        Returns:
        feature_importance: DataFrame with the feature importance of the best model
        '''

        # Get the stats from the GridSearchCV
        stats = pd.DataFrame.from_dict(grid.cv_results_)

        # Get the best model from the GridSearchCV
        best_model = grid.best_estimator_

        # Get the best parameters from the best model
        best_params = grid.best_params_

        # Get the index of the stats dataframe that have metrics different from None
        idx = stats.index[stats.iloc[:,0] != None]
        for key, value in best_params.items():
                # Iterate over the keys that have model hyperparameters (Discard the feature_selection hyperparameters)
                if 'model' in key:
                        idx = idx.intersection(stats.loc[:, f'param_{key}'].index[stats.loc[:, f'param_{key}'] == value])
                        
                # Get the slice of stats that contains the best parameters and iterates over the feature_selection hyperparameters
                best_model_stats = stats.loc[idx, :]


        # Check for the metric that ranked the best model     
        metric_ranking = grid.refit

        # Get the metric name that ranked the best model
        if isinstance(metric_ranking, bool):
            if metric_ranking:
                model_score = 'mean_test_score'
                model_rank = 'rank_test_score'
                model_std = 'std_test_score'
            else:
                model_score = None
        elif isinstance(metric_ranking, str):
            model_score = f'mean_test_{metric_ranking}'
            model_rank = f'rank_test_{metric_ranking}'
            model_std = f'std_test_{metric_ranking}'

        # Fetch the stats of the best model
        
        selector = best_model.named_steps['feature_selection']
        
        feature_score = selector.scores_
        feature_names = selector.feature_names_in_
        
        #MANU
        selected_features_mask = selector.get_support()
        selected_features = feature_names[selected_features_mask]
        
        print(colored(f'\nOriginal features before Feature Selection: {selector.feature_names_in_.tolist()}', 'green'))
        print(colored(f'\nSelected features after Feature Selection: {selected_features.tolist()}', 'green'))

        model_score = best_model_stats.loc[:, model_score]
        model_rank = best_model_stats.loc[:, model_rank]
        model_std = best_model_stats.loc[:, model_std]

        # Get the importance of the features
        feature_importance = pd.DataFrame(data=[feature_names, feature_score, selected_features_mask], index=['Feature', 'Score', 'Selected_feature']).T.sort_values(by='Score', ascending=False).reset_index(drop=True)
        feature_importance['model_score'] = model_score.values
        feature_importance['model_std'] = model_std.values
        feature_importance['model_rank'] = model_rank.values
        # Return the feature importance
        return feature_importance
    
    @staticmethod
    def get_cv_results(grid, dict = False):
        '''
        This method returns the cross-validation results of the model.

        Inputs:
        grid: GridSearchCV, fit grid object
        dict: bool, if True, return the cross-validation results as a dictionary

        Returns:
        cv_results: DataFrame or dictionary with the cross-validation results
        '''

        # Get the cross-validation results
        if not dict:
            cv_results = pd.DataFrame(grid.cv_results_)
        else:
            cv_results = grid.cv_results_

        # Return the cross-validation results
        return cv_results
    
    @staticmethod
    def AssessModel(model, X_test, y_test, predictIter = 100):
        '''
        This method evaluates the model performance through different metrics. The method uses the AssessModelRegression and AssessModelClassification methods to estimate the metrics.

        Inputs:
        model: model, model object
        X_test: DataFrame with the features of the testing set
        y_test: DataFrame with the target of the testing set
        predictIter: int, number of iterations to estimate the prediction time

        Outputs:
        eval: dict, dictionary with the metrics
        '''

        # Check if the model is not None, if not, assess the model
        if model is not None:
            # Infer the type of ml model used
            ml_type = Functions.get_type_ml_algorithm(model)


            # Assess the model performance
            if ml_type == 'regression':
                eval = Functions.AssessModelRegression(model, X_test, y_test)
            elif ml_type == 'classification':
                eval = Functions.AssessModelClassification(model, X_test, y_test)
            else:
                eval = {}

            # Estimate the prediction time 
            predictTime = Functions.EvalPredictionTime(model, X_test, predictIter)

            # Add the prediction time to the dictionary
            eval['pTime'] = predictTime

            # Return the dictionary with the metrics
            return eval
        
    @ staticmethod
    def get_model_stats(algorithm: str, target: str, ml_mode: str, feature_importance: pd.DataFrame|None, cv_results: pd.DataFrame, model_evaluation: dict|None, best_params: dict, tTime: float = 0.0):
        '''
        This method creates a dictionary with the stats of the model.

        Inputs:
        algorithm: str, algorithm name
        target: str, target variable
        ml_mode: str, machine learning mode
        feature_importance: DataFrame with the feature importance of the best model
        cv_results: DataFrame with the cross-validation results

        Outputs:
        stats: dict, dictionary with the stats of the model
        '''

        # Initialize the stats dictionary
        stats = {}

        # Append the training time to the model_evaluation dictionary
        if model_evaluation:
            model_evaluation['tTime'] = tTime

        # Create a dictionary for the model, if feature_importance is not None
        if feature_importance is not None:
            stats = {
                    'KQI': target,
                    'ml_mode': ml_mode,
                    'feature importance': feature_importance.to_dict(), 
                    'cv_results': cv_results, 
                    'scores': model_evaluation,
                    'model': best_params}
        else:
            stats = {
                    'KQI': target,
                    'ml_mode': ml_mode, 
                    'cv_results': cv_results, 
                    'scores': model_evaluation,
                    'model': best_params}
        
        return stats
    
    @staticmethod
    def save_ml_model(model_obj, name: str, target: str, ml_mode: str, feat_eng_mode: str, use_kqi: bool = False, format: str = 'joblib', path: str = './models'):
        '''
        This method saves a machine learning model in a file in format joblib. The model is saved in a directory that is named in function of the machine learning mode and feature engineering mode.

        Inputs:
        model_obj: model, model object to save
        name: str, name of the ml algorithm
        ml_mode: str, machine learning mode
        feat_eng_mode: str, feature engineering mode
        use_kqi: bool, if True, use the KQI to name the model
        path: str, path to the directory where the model will be saved

        Returns:
        None
        '''

        # Check if the path exists
        if Functions.CheckDirectory(path, msg=False):
            # Create a new directory for the models with the today's date inside the data directory
            today = pd.to_datetime('today').strftime('%Y-%m-%d')
            Functions.CheckDirectory(f'{path}/{today}', msg=False)

        # Define the pathname
        feat_eng_mode = Functions.map_feature_eng_mode(feat_eng_mode)

        # If use_kqi is True, use the KQI to name the model
        if use_kqi:
            fileName = f'{path}/{today}/{name}_{target}_{ml_mode}_{feat_eng_mode}_WKQI.{format}'
        else:
            fileName = f'{path}/{today}/{name}_{target}_{ml_mode}_{feat_eng_mode}_NoKQI.{format}'

        # Save the model
        Functions.SaveModel(model_obj, fileName)
    
    
    #FUNCIONES PARA "TrainGridSearchCV"
    
    @staticmethod
    def get_type_ml_algorithm(pipe):
        '''
        This method returns the type of machine learning model used in the pipeline.

        Inputs:
        pipe: pipeline, pipeline object
        
        Returns:
        ml_type: str, type of machine learning model (regression, classification)
        '''
        
        # Infer the type of ml model used
        if 'Regressor' in pipe.named_steps['model'].__class__.__name__:
            ml_type = 'regression'
        elif 'Classifier' in pipe.named_steps['model'].__class__.__name__:
            ml_type = 'classification'
        elif 'Ridge' in pipe.named_steps['model'].__class__.__name__:
            ml_type = 'regression'
        elif 'SVR' in pipe.named_steps['model'].__class__.__name__:
            ml_type = 'regression'
        elif 'SVC' in pipe.named_steps['model'].__class__.__name__:
            ml_type = 'classification'
        elif 'GaussianNB' in pipe.named_steps['model'].__class__.__name__:
            ml_type = 'classification'
        else:
            ml_type = None

        # Return the type of ml model
        return ml_type
    
    @staticmethod
    def mean_average_error_percentage(y_true, y_pred):
        '''
        This method calculates the mean average error percentaje of the model. The error value is not in scale [0, 100]. 0 means no error and 1 means 100% error with respect to the mean of y_true.

        Inputs:
        y_true: DataFrame/array with the true target
        y_pred: DataFrame/array with the predicted target

        Returns:
        error
        '''
        
        # Calculate the mean average error percentaje
        error = (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))

        # Return the error
        return error
    
    
    #FUNCIONES PARA "AssessModel"
    
    @staticmethod
    def AssessModelRegression(model, X_test, y_test):
        '''
        This method evaluates the model performance through different metrics.
        
        Inputs:
        model: model, model object
        X_test: DataFrame with the features of the testing set
        y_test: DataFrame with the target of the testing set

        Outputs:
        eval: dict, dictionary with the metrics
        '''

        # Predict the target 
        y_pred = model.predict(X_test)

        # Calculate the mean value of the predicted target
        y_avg = y_test.mean()

        # Estimate the metrics
        R2 = r2_score(y_test, y_pred)

        MAE = mean_absolute_error(y_test, y_pred)

        MSE = mean_squared_error(y_test, y_pred)

        MAPE = mean_absolute_percentage_error(y_test, y_pred)
   
        MedAE = median_absolute_error(y_test, y_pred)
 
        EVS = explained_variance_score(y_test, y_pred)

        MAEP = MAE / y_avg

        RMSE = math.sqrt(MSE)

        RMSEP = RMSE / y_avg

        # Return a dictionary with the metrics
        report = {'R2': R2, 'MAE': MAE, 'MSE': MSE, 'MAPE': MAPE, 'MedAE': MedAE, 'EVS': EVS, 'MAEP': MAEP, 'RMSE': RMSE, 'RMSEP': RMSEP}
        return report

    @staticmethod
    def AssessModelClassification(model, X_test, y_test):
        
        '''
        This method evaluates the model performance through different metrics. This method uses the classification_report method from the sklearn.metrics package.

        Inputs:
        model: model, model object
        X_test: DataFrame with the features of the testing set
        y_test: DataFrame with the target of the testing set

        Outputs:
        report: dict, dictionary with the metrics
        '''
        # Predict the target
        y_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            y_pred_prob = model.predict_proba(X_test)
            
            # Estimate the metrics
            AUC = roc_auc_score(y_test, y_pred_prob[:,1], multi_class='ovr', average='weighted')
            log = log_loss(y_test, y_pred_prob)
        else:
            AUC = None
            log = None

        # Estimate the other metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)

        recall = recall_score(y_test, y_pred, average='weighted')

        f1_micro = f1_score(y_test, y_pred, average='micro')

        f1_macro = f1_score(y_test, y_pred, average='macro')

        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        mcc = matthews_corrcoef(y_test, y_pred)

        # Return a dictionary with the metrics
        report = {'AUC': AUC, 
                  'Accuracy': accuracy, 
                  'Precision': precision, 'Recall': recall, 
                  'F1_micro': f1_micro, 
                  'F1_macro': f1_macro, 
                  'F1_weighted': f1_weighted, 
                  'Log_Loss': log, 
                  'MCC': mcc}
        return report
    
    @staticmethod
    def EvalPredictionTime(model, X_test, predictIter = 100):
        '''
        This method evaluates the prediction time of the model.

        Inputs:
        model: model, model object
        X_test: DataFrame with the features of the testing set
        predictIter: int, number of iterations to estimate the prediction time

        Outputs:
        predictTime: float, prediction time
        '''

        # Initialize the list of times
        times = []
        
        # Iterate over the number of iterations to estimate the prediction time
        for i in range(0, predictIter):
            # Get a sample from the testing set
            x_sample = X_test.sample()
            idx = x_sample.index[0]
            
            # Estimate the prediction time
            startTime = timer()

            y_est = model.predict(x_sample)

            #Training elapsed time
            finishTime = timer()
            predictTime = finishTime - startTime
            times.append(predictTime)

        # Calculate the average prediction time among the iterations
        predictTime = pd.DataFrame(times).mean().values[0]

        # Return the prediction time
        return predictTime
    
    
    #FUNCIONES PARA "save_ml_model"
    
    @staticmethod
    def CheckDirectory(path, msg: bool = False):
        '''
        This method check if a directory exists. If the directory does not exist, the method creates the directory.
        
        Inputs:
        path: str, path to the directory
        msg: bool, if True, print confirmation messages
        
        Returns:
        True, if the directory exists
        False, if the directory does not exist
        '''

        #Check if the path exists
        if not os.path.exists(path):
            os.makedirs(path)
            # If not, create and double check if the directory was created
            if Functions.CheckDirectory(path, msg=False):
                if msg: print(f'Directory {path} created')
                return True
        else:
            if msg: print(f'Directory {path} already exists')
            return True
        
        # If was not possible to find or create the directory, return False
        return False
    
    @staticmethod
    def map_feature_eng_mode(feat_eng_mode: str|list):
        '''
        This methods maps the feature engineering mode to a short version. The short version is used to name the models and the files.

        Inputs:
        feat_eng_mode: str, feature engineering mode to map

        Returns:
        str, short version of the feature engineering mode
        '''

        # If feat_eng_mode is 'FS1', return 'FS'
        if feat_eng_mode == 'Feature_selection':
            return 'FS'
        # If feat_eng_mode is 'PCA', return 'PCA'
        elif feat_eng_mode == 'Feature_extraction':
            return 'FE'
        # If feat_eng_mode is 'No_FE', return 'No_FE'
        elif feat_eng_mode == 'No_FE':
            return 'No_FE'
        # If string is empty, return 'No_FE'
        else:
            return 'No_FE'
        
    @staticmethod
    def SaveModel(model, pathname):
        '''
        This method saves a model in a file in format joblib.

        Inputs:
        model: model, model to save
        pathname: str, path to the file to save

        Returns:
        None
        '''

        # Import the necessary packages
        from joblib import dump

        # Save the model in a file
        dump(model, pathname)

        # Print confirmation
        print(colored(f'Model saved as: {pathname}\n', 'yellow', attrs=['bold']))
        
        
    #FUNCIONES PARA EL DATASET

    @staticmethod
    def RandomizeDataSet(dataInput, shuffle = True):
        '''
        This method randomizes the dataset. The dataset is randomized through the rows.

        Inputs:
        dataInput: DataFrame with the features of the dataset
        do: bool, if True, randomize the dataset    

        Returns:  
        dataInput: DataFrame with the randomized dataset
        '''

        #Randomize the dataSet through the rows
        if shuffle:
            dataInput = dataInput.sample(frac=1)

        return dataInput
        
    @staticmethod
    def ApplyPCA(data, nComp = 1):
        '''
        This method applies PCA to the dataset. The PCA is used to reduce the number of  features in the dataset.
        
        Inputs:
        data: DataFrame with the features of the dataset
        nComp: int, number of components to reduce the dataset

        Returns:
        X: DataFrame with the PCA transformed dataset
        '''

        # Determine the data size and save the number of columns (features)
        size = data.shape
        nFeatures = size[1]

        #If nComp = -1 do not do PCA transformation
        if nComp == -1:
            X = data
        # Else, do PCA transformation with nComp
        else:
            if nComp <= nFeatures:
                pca = PCA(n_components = nComp)
                pca.fit(data)
            else:
                pca = PCA(n_components = 1)
                pca.fit(data)

            X = pca.transform(data)

        return X
    
    @staticmethod
    def create_FE_FS_dataset(X: pd.DataFrame, Y: pd.DataFrame, feature_modes: list = [], scale: bool = True, save_dataset: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        This method creates the feature engineering and feature selection datasets. 
        For FS: The method removes the multicollinearity and the skewness of the dataset. 
        By default, the method also applies the scaler to the dataset.

        Inputs:
        X: DataFrame with the features of the dataset
        Y: DataFrame with the target of the dataset
        feature_modes: list, list of feature engineering modes
        scale: bool, if True, apply scale to the features
        save_dataset: bool, if True, save the dataset

        Returns:
        X_train_FE: DataFrame with the features of the feature engineering dataset
        X_test_FE: DataFrame with the features of the feature engineering dataset
        X_train_FS: DataFrame with the features of the feature selection dataset
        X_test_FS: DataFrame with the features of the feature selection dataset
        '''
        
        # Split dataset and save the split
        (X_train, y_train, X_test, y_test) = Functions.SplitDataset(X=X, y=Y, scale=False, save=save_dataset)

        # If needed, apply scaler
        if scale:
            X_train, X_test = Functions.ApplyScaler(X_train=X_train, X_test=X_test, save_scaler=True)

        # If feature_selection is present in the configuration, apply it
        if 'Feature_selection' in feature_modes:
            # Remove features with high VIF
            dataset_FS, vif_df = Functions.RemoveMulticollinearity(X_train=X_train, X_test=X_test, threshold=5, save_multicollinearity=True)

            # Remove skewed features
            dataset_FS = Functions.RemoveSkewness(dataset_FS, 1.0, save_skewness=True)
            
            # Extract the scaled/non-scaled features according to the valid features
            X_train_FS = dataset_FS.loc[X_train.index, dataset_FS.columns]
            X_test_FS = dataset_FS.loc[X_test.index, dataset_FS.columns]

            # Save the FS dataset
            if save_dataset:
                Functions.SaveTrainTestSet(X_train=X_train_FS, y_train=y_train, X_test=X_test_FS, y_test=y_test, name='FS')
        else:
            X_train_FS = pd.DataFrame()
            X_test_FS = pd.DataFrame()

        # Copy the original scaled/non-scaled dataset to FE dataset
        X_train_FE = X_train.copy()
        X_test_FE = X_test.copy()

        return X_train_FE, X_test_FE, X_train_FS, X_test_FS, y_train, y_test
    
    @staticmethod
    def SplitDataset(X: pd.DataFrame, y: pd.DataFrame, random_state: int = 0, scale: bool = False, save: bool = True):
        '''This method splits the dataset into a training and testing set. Then save the split datasets for future usage (validation, testing, ...)
        
        Inputs:
        X: DataFrame with the features of the dataset
        y: DataFrame with the target of the dataset
        random_state: int, random state for the split
        scale: bool, if True, apply scale to the features
        
        Returns:
        X_train: DataFrame with the features of the training set
        y_train: DataFrame with the target of the training set
        X_test: DataFrame with the features of the testing set
        y_test: DataFrame with the target of the testing set
        '''

        #Split the dataset into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= random_state)

        # If scale is selected, apply scale to the features
        if scale:
            (X_train, X_test) = Functions.ApplyScaler(X_train, X_test)
        else:
            # Regenerate Xvalues as DataFrames
            headers = X.columns
            X_train = pd.DataFrame(data=X_train, columns = headers)
            X_test = pd.DataFrame(data=X_test, columns = headers)

        # Save the split dataset 
        if save:
            Functions.SaveTrainTestSet(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # Return the split datasets
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def ApplyScaler(X_train: pd.DataFrame, X_test: pd.DataFrame, save_scaler: bool = False):
        '''
        This method scales the dataset. The dataset is scaled to have a mean of 0 and a standard deviation of 1. The scaling is applied to the training and testing set.
        
        Inputs:
        X_train: DataFrame with the features of the training set
        X_test: DataFrame with the features of the testing set
        
        Returns:
        X_trainSc: DataFrame with the scaled features of the training set
        X_testSc: DataFrame with the scaled features of the testing set
        '''

        # Apply feature standardization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # Fit the scaler only within the train data
        scaler.fit(X_train)

        # Transform the data
        X_trainSc = scaler.transform(X_train)
        X_testSc = scaler.transform(X_test)
        
        # Format the data as a DataFrame
        X_train_scaled = pd.DataFrame.from_records(X_trainSc, columns=X_train.columns)
        X_test_scaled = pd.DataFrame.from_records(X_testSc, columns=X_test.columns)

        # Save scaler
        if save_scaler:
            Functions.SaveScaler(scaler)
        # Return the scaled datasets in format DataFrame
        return X_train_scaled, X_test_scaled
    
    @staticmethod
    def SaveScaler(scaler: StandardScaler, path='./data', name='stdScaler'):
        '''
        This method saves the scaler for future usage (estimating, testing, ...)
        
        Inputs:
        scaler: StandardScaler used to preprocess data
        path: str, path to the directory where the scaler will be saved (Be careful with the path)
        
        Returns:
        None
        '''
        # Check if the data path exists
        if Functions.CheckDirectory(path, msg=True):
            # Create a new directory for the split datasets with the today's date inside the data directory
            today = pd.to_datetime('today').strftime('%Y-%m-%d')
            Functions.CheckDirectory(f'{path}/{today}', msg=True)

        # Format the data as json and save it to a file
        fileName = f'{path}/{today}/{name}.pkl'

        joblib.dump(scaler, fileName)
            
        print(colored(f'Scaler have been saved in: {fileName}', 'green', attrs=['bold']))

    @staticmethod
    def LoadScaler(name: str):
        '''
        This method loads a pkl file and returns the correspondand scaler.

        Inputs:
        name: str, name of the file to load without the extension

        Returns:
        scaler: StandardScaler, StandardScaler of the pkl file
        '''
        
        fileName = '{}.pkl'.format(name)
        
        scaler = joblib.load(fileName)
            
        return scaler
    
    @staticmethod
    def RemoveMulticollinearity(X_train: pd.DataFrame, X_test:pd.DataFrame, threshold = 5, save_multicollinearity: bool = False):
        '''
        This method cleans the dataset by removing the columns that have a VIF higher than the threshold. (Remove multicollinearity)
        
        Inputs:
        X_train: DataFrame with the features of the training set
        X_test: DataFrame with the features of the testing set
        threshold: float, threshold to remove multicollinearity, VIF >=5

        Returns:
        data: DataFrame with the features of the dataset without multicollinearity
        '''
               
        #Concat scaled/non-scaled training and test dataframes
        data = pd.concat([X_train, X_test], axis=0)
        data = data.reset_index(drop=True)
        
        # Calculate the MI matrix of the input
        print('Calculating MI matrix between the inputs')
        mi_df = Functions.GetMIMatrix_Input(data)

        # Iteratively check if there are any infinity values in the VIF matrix
        ban = True
        print('Checking multicollinearity through VIF values')
        print('Deleting deatures with VIF values equal to infinity')
        while(ban):
            # Calculate the VIF matrix of the input
            vif_df = Functions.CalculateVIF(data)

            # Check for VIF values equal than infinity
            if len(vif_df[vif_df['VIF'] == np.inf]) > 0:
                inf_keys = vif_df[vif_df['VIF'] == np.inf].index.tolist()
                print(f'\t{inf_keys}')
                # Find the first occurence of the inf value list in the MI dataframe
                delete_ft = (mi_df.sum().sort_values(ascending = False))[inf_keys].idxmax()
                data = data.drop(columns = delete_ft, inplace = False)
            else:
                ban = False

        # Delete the feature with the highest VIF value until a threshold is reached
        print(f'\nDeleting features with VIF values higher than the threshold = {threshold}')
        while vif_df['VIF'].max() > threshold: 
            # Find the max VIF value in the dataframe and drop that feature
            delete_ft = vif_df.idxmax()
            print(colored(f'\tVIF {delete_ft.values} = {vif_df.loc[delete_ft, "VIF"].values[0]}', 'red'))
            data = data.drop(columns = delete_ft, inplace = False)
            # Recalculate the VIF values
            vif_df = Functions.CalculateVIF(data)
            
        # Print in green color "the selected features"
        print(colored(f'\nSelected features: {data.columns.tolist()}', 'green'))
        
        # Save "the selected features" for new data
        if save_multicollinearity:
            Functions.SaveMulticollinearity(data.columns.tolist())
        
        # Return the cleaned dataset and the VIF dataframe
        return data, vif_df

    @staticmethod
    def SaveMulticollinearity(selected_features: list[str], path='./data', name='multicollinearitySelectedFeats',):
        '''
        This method saves the selected features that have a VIF lower than the threshold. (Remove multicollinearity)
        
        Inputs:
        selected_features: list[str], selected features to remove multicollinearity
        path: str, path to the directory where the list will be saved (Be careful with the path)
        
        Returns:
        None
        '''
        # Check if the data path exists
        if Functions.CheckDirectory(path, msg=True):
            # Create a new directory for the split datasets with the today's date inside the data directory
            today = pd.to_datetime('today').strftime('%Y-%m-%d')
            Functions.CheckDirectory(f'{path}/{today}', msg=True)

        # Format the data as json and save it to a file
        fileName = f'{path}/{today}/{name}.json'

        with open(fileName, "w") as file:
            json.dump(selected_features, file, indent=4)

        print(colored(f'Selected features for removing multicollinearity have been saved in: {fileName}', 'green', attrs=['bold']))

    @staticmethod
    def LoadMulticollinearity(name: str):
        '''
        This method loads a json file and returns the selected features to remove multicollinearity.

        Inputs:
        name: str, name of the file to load without the extension

        Returns:
        selected_features: list[str], selected features to remove multicollinearity
        '''
        
        fileName = '{}.json'.format(name)
        
        with open(fileName, "r") as file:
            selected_features = json.load(file)
            
        return selected_features
    
    @staticmethod
    def RemoveSkewness(data, threshold = 1.0, save_skewness: bool = False):
        '''
        This method remove skewness from the dataset. The skewness is removed by applying the QuantileTransformer to the dataset.
        
        Inputs:
        data: DataFrame with the features of the dataset
        threshold: float, threshold to remove skewness

        Returns:
        data: DataFrame with the features of the dataset without skewness

        '''
        # Import the necessary packages
        from scipy.stats import skew
        from sklearn.preprocessing import QuantileTransformer

        # Determine the skewness of the dataset
        skewness = skew(data) 

        # Get the skewness variables of the dataset with skewness greater than 1 or less than -1
        cond = (skewness < -threshold) | (skewness > threshold)

        # Get the columns with skewness greater than 1 or less than -1
        skewclasses = data[data.columns[cond]]

        # Train the QuantileTransformer
        quantile = QuantileTransformer(output_distribution='normal', random_state=0)

        # Fit and transform the dataset
        skewclasses = pd.DataFrame(data=quantile.fit_transform(skewclasses), columns=data.columns[cond])

        # Replace the skewed columns with the transformed columns
        data[skewclasses.columns] = skewclasses

        # Save QuantileTransformer and "the columns transformed" for new data
        if save_skewness:
            Functions.SaveSkewness(quantile, data.columns[cond])
            
        # Return the transformed dataset
        return data
    
    @staticmethod
    def SaveSkewness(quantileTransformer: QuantileTransformer, skewed_columns: list[str], path='./data', name_transformer='quantileTransformer', name_columns='skewedColumns'):
        '''
        This method saves the QuantileTransformer and the list of skewed columns applied to the dataset for removing skewness
        
        Inputs:
        quantileTransformer: QuantileTransformer used to preprocess data
        name_transformer: str, name for QuantileTransformer file
        skewed_columns: list[str], skewed columns to get transformed
        name_columns: str, name for list of skewed columns
        path: str, path to the directory where the transformer will be saved (Be careful with the path)
        
        Returns:
        None
        '''
        # Check if the data path exists
        if Functions.CheckDirectory(path, msg=True):
            # Create a new directory for the split datasets with the today's date inside the data directory
            today = pd.to_datetime('today').strftime('%Y-%m-%d')
            Functions.CheckDirectory(f'{path}/{today}', msg=True)

        # Format the data as pickle and save it to a file
        fileName = f'{path}/{today}/{name_transformer}.pkl'

        joblib.dump(quantileTransformer, fileName)

        print(colored(f'QuantileTransformer have been saved in: {fileName}', 'green', attrs=['bold']))
        
        # Format the data as json and save it to a file
        fileName = f'{path}/{today}/{name_columns}.json'

        with open(fileName, "w") as file:
            json.dump(skewed_columns.tolist(), file, indent=4)

        print(colored(f'Skewed columns have been saved in: {fileName}', 'green', attrs=['bold']))

    @staticmethod
    def LoadSkewness(name_transformer: str, name_columns: str):
        '''
        This method loads a pkl file and a json file and returns the correspondand quantile transformer and the skewed columns to get transformed.

        Inputs:
        name_transformer: str, name of the file to load without the extension
        name_columns: str, name of the file to load without the extension
        
        Returns:
        quantileTransformer: QuantileTransformer used to preprocess data
        selected_features: list[str], selected features to remove multicollinearity
        '''
        
        '''
        This method loads a json file and returns the selected features to remove multicollinearity.

        Inputs:
        name: str, name of the file to load without the extension

        Returns:
        skewed_columns: list[str], skewed columns to get transformed
        '''
        
        fileName = '{}.pkl'.format(name_transformer)
        
        quantileTransformer = joblib.load(fileName)
            
        fileName = '{}.json'.format(name_columns)
        
        with open(fileName, "r") as file:
            skewed_columns = json.load(file)
            
        return quantileTransformer, skewed_columns
    
    @staticmethod
    def SaveTrainTestSet(X_train: pd.DataFrame = pd.DataFrame(), y_train: pd.DataFrame = pd.DataFrame(), X_test: pd.DataFrame = pd.DataFrame(), y_test: pd.DataFrame = pd.DataFrame(), path='./data', name='NoFS'):
        '''
        This method saves the split datasets for future usage (validation, testing, ...)
        
        Inputs:
        X_train: DataFrame with the features of the training set
        y_train: DataFrame with the target of the training set
        X_test: DataFrame with the features of the testing set
        y_test: DataFrame with the target of the testing set
        path: str, path to the directory where the split dataset will be saved (Be careful with the path)
        
        Returns:
        None
        '''

        # Check if the data path exists
        if Functions.CheckDirectory(path, msg=True):
            # Create a new directory for the split datasets with the today's date inside the data directory
            today = pd.to_datetime('today').strftime('%Y-%m-%d')
            Functions.CheckDirectory(f'{path}/{today}', msg=True)

        # Append the datasets into a single dictionary
        data = {
                'X_train': X_train.to_dict(),
                'y_train': y_train.to_dict(),
                'X_test': X_test.to_dict(),
                'y_test': y_test.to_dict()
        }

        # Format the data as json and save it to a file
        Functions.SaveJson(data, f'{path}/{today}/{name}_split_dataset')
    
    @staticmethod
    # This method reads the split dataset in format parquet.gzip
    def ReadTrainTestSet(path='./data'):
        '''
        This method reads the split dataset in format json and returns the split datasets
        
        Inputs:
        path: str, path to the directory where the split dataset is located (Be careful with the path)
        
        Returns:
        X_train: DataFrame with the features of the training set
        y_train: DataFrame with the target of the training set
        X_test: DataFrame with the features of the testing set
        y_test: DataFrame with the target of the testing set
        '''

        # Read the json file
        data = Functions.LoadJson(path)

        # Split the loaded dataset in json into X_train, y_train, X_test, y_test
        X_train = pd.DataFrame.from_dict(data.get('X_train'), orient='columns')
        y_train = pd.DataFrame.from_dict(data.get('y_train'), orient='columns')
        X_test = pd.DataFrame.from_dict(data.get('X_test'), orient='columns')
        y_test = pd.DataFrame.from_dict(data.get('y_test'), orient='columns')

        # Return the split datasets
        return X_train, y_train, X_test, y_test
    
    #FUNCIONES SCORE
    @staticmethod
    def get_scores(targets: list, feature_modes: list, scale: bool = True, path: str = './stats_models/2024-03-21/'):
        '''
        This method gets the scores of the models. The scores are extracted from the json files that are saved in the stats_models directory.

        Inputs:
        targets: list, list of targets
        feature_modes: list, list of feature engineering modes
        scale: bool, if True, scale the scores
        path: str, path to the directory where the stats are saved

        Returns:
        scores: DataFrame with the scores of the models
        '''

        # Initialize the dataframe to store the scores
        scores = pd.DataFrame()
        counter = 0

        # Iterate over the targets
        for target in targets:
            stats = pd.read_json(f'{path}stats_{target}.json')
            # Iterate over the feature modes
            for feat_mode in feature_modes:
                # Get the algorithms available in the dictionary
                algorithms = list(stats.loc[feat_mode].loc[target].keys())
                # Get the stats per feature mode
                stats_fm = stats.loc[feat_mode]

                # Create a dataframe with the scores per algorithm
                for algorithm in algorithms:
                    # If algorithm is regression, get the MAEP, otherwise get the F1_weighted
                    ml_mode = stats_fm.loc[target].get(algorithm).get('ml_mode')
                    if ml_mode == 'regression':
                        metric = 'MAEP'
                        #If scale is used, get the scores and scale them
                        if scale:
                            scale_feat = True
                        else:
                            scale_feat = False
                    else:
                        metric = 'F1_weighted'
                        scale_feat = False
                    
                    if feat_mode == "FS_NoPreprocessed":
                        feat_mode = "FS_NoPre"
                    elif feat_mode == "FS_Preprocessed":
                        feat_mode = "FS_Pre"
                    # Get the scores
                    if scale_feat:
                        scores[counter] = {'algorithm': algorithm if ml_mode == 'regression' else Functions.map_ml_algorithm_labels(algorithm), 
                                           'value': Functions.FixDataScale(data=float(stats_fm.loc[target].get(algorithm).get('scores').get(metric)), metric=metric), 
                                           'metric': metric, 
                                           'pTime': Functions.FixDataScale(data=float(stats_fm.loc[target].get(algorithm).get('scores').get('pTime')), metric='pTime'), 
                                           'feat_mode': feat_mode, 
                                           'target': target, 
                                           'ml_mode': ml_mode}
                    else:
                        scores[counter] = {'algorithm': algorithm if ml_mode == 'regression' else Functions.map_ml_algorithm_labels(algorithm),
                                           'value': float(stats_fm.loc[target].get(algorithm).get('scores').get(metric)), 
                                           'metric': metric, 
                                           'pTime': Functions.FixDataScale(data=float(stats_fm.loc[target].get(algorithm).get('scores').get('pTime')), metric='pTime'), 
                                           'feat_mode': feat_mode, 
                                           'target': target, 
                                           'ml_mode': ml_mode}

                    # Increase the counter
                    counter += 1

        # Transpose the dataframe
        scores = scores.T 

        # Return the scores
        return scores
    
    @staticmethod
    def get_feature_scores(targets: list, path: str = './stats_models/2024-03-21/'):
        '''
        This method gets the feature scores of the models. The feature scores are extracted from the json files that are saved in the stats_models directory.
        This function only works for Feature Selection.

        Inputs:
        targets: list, list of targets
        path: str, path to the directory where the stats are saved

        Returns:
        feature_scores: DataFrame with the feature scores of the models
        '''
        
        # Initialize the feature scores dataframe
        feature_scores = pd.DataFrame()

        # Iterate over the targets
        for target in targets:
            # Read the stats file
            stats = pd.read_json(f'{path}stats_{target}.json')

            # Extract the stats per target
            data = stats.loc['FS'].get(target)

            # Get the available algorithms
            if data:
                algorithms = list(data.keys())
            else:
                algorithms = []
            
            # Initialize the feature data
            feature_data = pd.DataFrame()
            
            # Iterate over the algorithms
            for algorithm in algorithms:
                if data:
                    feature_importance = data.get(algorithm).get('feature importance')

                if feature_importance:
                    # Read the feature importance data from the stats
                    feature_importance = pd.DataFrame.from_dict(feature_importance)

                    #Append the target name and the algorithm name
                    feature_importance['algorithm'] = algorithm
                    feature_importance['target'] = target
                    feature_importance['ml_mode'] = data.get(algorithm).get('ml_mode') # type: ignore
                    

                    # Get the absoluite value of column model_score
                    feature_importance['model_score'] = feature_importance['model_score'].abs()

                # Append the feature data per algorithm to the feature data per target
                feature_data = pd.concat([feature_data, feature_importance], axis = 0)

            # Append the feature data per target to the overall feature data
            feature_scores = pd.concat([feature_scores, feature_data], axis = 0)

        # Return the feature scores
        return feature_scores
    
    @staticmethod
    def estimate_pet_score(scores: pd.DataFrame, pTime_norm_factor: float|str = 'max', pTime_weight: float = 0.5):
        '''
        This method estimates the PET score of the models. The PET score is a metric that combines the model performance value and the prediction time.
        
        Inputs:
        scores: DataFrame with the scores of the models
        pTime_norm_factor: float, normalization factor for the prediction time
        pTime_weight: float, weight of the prediction time in the PET score

        Returns:
        scores: DataFrame with the PET score of the models
        '''

        # If pTime_norm factor is 'max', normalize the prediction time by the maximum value, elif other options are available, normalize by the mean, median or min
        if pTime_norm_factor == 'max':
            pTime_norm_factor = scores.pTime.max()
        elif pTime_norm_factor == 'mean':
            pTime_norm_factor = scores.pTime.mean()
        elif pTime_norm_factor == 'median':
            pTime_norm_factor = scores.pTime.median()
        elif pTime_norm_factor == 'min':
            pTime_norm_factor = scores.pTime.min()
        elif pTime_norm_factor == 'quantile25':
            pTime_norm_factor = scores.pTime.quantile(0.25)
        elif pTime_norm_factor == 'quantile50':
            pTime_norm_factor = scores.pTime.quantile(0.50)
        elif pTime_norm_factor == 'quantile75':
            pTime_norm_factor = scores.pTime.quantile(0.75)
        elif pTime_norm_factor == 'quantile90':
            pTime_norm_factor = scores.pTime.quantile(0.90)
        elif pTime_norm_factor == 'quantile95':
            pTime_norm_factor = scores.pTime.quantile(0.95)
        elif pTime_norm_factor == 'quantile99':
            pTime_norm_factor = scores.pTime.quantile(0.99)
        
        # If not, use the value passed
        else: 
            pTime_norm_factor = pTime_norm_factor

        # Normalize the prediction time by a factor (Default: 10ms which is the max time for Near-real-time RIC)
        scores['n_pTime'] = (scores.pTime / pTime_norm_factor)

        # For regression problems, the PET Score is a function of the MAEP (error metric) and the pTime
        r_scores = scores[scores.ml_mode == 'regression']
        # For classification problems, the PET Score is a function of the F1_score and the pTime
        c_scores = scores[scores.ml_mode == 'classification']

        # Get the inverse of the pTime weight
        pTime_const = 1/pTime_weight
        error_const = (1 - pTime_weight)

        # Calculate the PET score
        #PET_score = pd.concat([2 * ((r_scores.n_pTime * (r_scores.value / 100)) / (r_scores.n_pTime + (r_scores.value / 100))), 
        #                    2 * ((c_scores.n_pTime * ((1 - c_scores.value)/100)) / (c_scores.n_pTime) + ((1 - c_scores.value)/100))], axis=0)
        PET_score = pd.concat([(r_scores.n_pTime * (r_scores.value / 100)) / (pTime_weight * ((r_scores.value / 100) - r_scores.n_pTime) + r_scores.n_pTime), 
                            (c_scores.n_pTime * (1 - (c_scores.value * 0.999))) / (pTime_weight * ((1 - (c_scores.value * 0.999)) - c_scores.n_pTime) + c_scores.n_pTime)], axis=0)

        # Add the PET score to the scores dataframe
        scores['PET_score'] = PET_score
        scores['comp_PET_score'] = 1 - PET_score

        # Return the scores
        return scores
    
    
    #FUNCIONES CORRELACION

    @staticmethod
    def get_mi_inputs_targets_baseline(targets: list, path: str = './data/2024-03-21/FS_split_dataset', only_features: bool = False):
        '''
        This method calculates the Mutual Information of the input features with the targets. The MI is calculated for the baseline dataset. This is the original MI thart every feature has with the target and other features in the split dataset.

        Inputs:
        targets: list, list of targets
        path: str, path to the directory where the split dataset is located

        Returns:
        MI: DataFrame with the Mutual Information of the input features with the targets
        '''
        
        # Read the split datasets
        X_train, y_train, _, _ = Functions.ReadTrainTestSet(path)

        # Reset the index
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)

        # Extract only the targets from the training set
        if only_features:
            info = X_train
        else:
            target_data = y_train[targets]
            info = pd.concat([X_train, target_data], axis=1)

        # For each target get the MI score
        MI = Functions.GetMIMatrix_Input(info)

        # Return the MI
        return MI
    
    
    @staticmethod
    def get_Nmi_inputs_targets_baseline(targets: list, path: str = './data/2024-03-21/FS_split_dataset'):
        '''
        This method calculates the Mutual Information of the input features with the targets. The MI is calculated for the baseline dataset. This is the original MI thart every feature has with the target and other features in the split dataset.

        Inputs:
        targets: list, list of targets
        path: str, path to the directory where the split dataset is located

        Returns:
        MI: DataFrame with the Mutual Information of the input features with the targets
        '''
        
        # Read the split datasets
        X_train, y_train, _, _ = Functions.ReadTrainTestSet(path)

        # Reset the index
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)

        # Extract only the targets from the training set
        target_data = y_train[targets]

        # For each target get the MI score
        info = pd.concat([X_train, target_data], axis=1)
        MI = Functions.GetNMIMatrix_Input(info)

        # Return the MI
        return MI
    
    @staticmethod
    def rename(columns):
         
        new_columns = []
        
        for column_name in columns:
            if column_name == "Reach1080p+":
                new_columns.append("AlcanzaHD")
            elif column_name == "timeToReach1080p+":
                new_columns.append("TiempoAlcanzarHD")
            elif column_name.startswith('Reach'):
                new_columns.append(column_name.replace('Reach', 'Alcanza'))
            elif column_name.startswith('timeToReach'):
                new_columns.append(column_name.replace('timeToReach', 'TiempoAlcanzar'))
            elif column_name in ['1080p', '2K', '4K']:
                new_columns.append('Porcentaje' + column_name)
            else:
                new_columns.append(column_name)
        return new_columns
    @staticmethod
    def GetMIMatrix_Input(data):
        '''
        This method returns the MI matrix of the input using the CalculateMI method iteritavely along all the input columns.
        
        Inputs:
        data: DataFrame with the features of the dataset

        Returns:
        mi_df: DataFrame with the MI matrix of the input
        '''

        # Extract the target iteratively as a column value from data and calculate the MI
        values = {}

        for column in data.columns:
            #X = data.drop(columns = column)
            X = data.copy()
            target = data[column]
            values.update({column : pd.Series(Functions.CalculateMI(X, target), index = X.columns, name = column)})

        # Generate a DataFrame with the MI values
        mi_df = pd.DataFrame(values)

        # Change NaN values to 0
        mi_df.fillna(np.inf, inplace = True)

        # Return the MI matrix
        return mi_df
    
    @staticmethod
    def CalculateMI(data, target):
        '''
        This method calculates the Mutual Information of the dataset.
        
        Inputs:
        data: DataFrame with the features of the dataset
        target: DataFrame with the target of the dataset

        Returns:
        mi: DataFrame with the Mutual Information of the dataset
        '''
        #Calculate the mutual information between the features and the target
        mi = mutual_info_regression(data, target, random_state=0, discrete_features=False)

        return mi
    
    @staticmethod
    def GetNMIMatrix_Input(data):
        '''
        This method returns the MI matrix of the input using the CalculateMI method iteritavely along all the input columns.
        
        Inputs:
        data: DataFrame with the features of the dataset

        Returns:
        mi_df: DataFrame with the MI matrix of the input
        '''

        # Extract the target iteratively as a column value from data and calculate the MI
        values = {}

        for column in data.columns:
            X = data.copy()
            target = data[column]
            mi_scores = []
            for feature in X.columns:
                mi_score = Functions.CalculateNMI(X[feature], target)
                mi_scores.append(mi_score)
            values[column] = pd.Series(mi_scores, index=X.columns, name=column)

        mi_df = pd.DataFrame(values)
        # Change NaN values to 0
        mi_df.fillna(np.inf, inplace = True)

        return mi_df
    
    @staticmethod
    def CalculateNMI(feature, target):
        '''
        This method calculates the Mutual Information of the dataset.
        
        Inputs:
        data: DataFrame with the features of the dataset
        target: DataFrame with the target of the dataset

        Returns:
        mi: DataFrame with the Mutual Information of the dataset
        '''
        # Ensure the feature and target are 1D arrays
        feature = feature.values.ravel()
        target = target.values.ravel()
        
        # Calculate the normalized mutual information
        nmi = normalized_mutual_info_score(feature, target)

        # Return the mutual information
        return nmi
    
    @staticmethod
    def CalculateVIF(data):
        '''
        This method calculates the VIF of the dataset. The VIF is used to determine the multicollinearity of the features in the dataset.
        
        Inputs:
        data: DataFrame with the features of the dataset

        Returns:
        vif_df: DataFrame with the VIF of the dataset
        '''
        #Using VIF (Variance Inflation Factor) to delete multicollinearity (>10 high dependance)
        vif_mat = [vif(data.values, i) for i in range(len(data.columns))]
        vif_df = pd.DataFrame(vif_mat, index = list(data.columns), columns = ['VIF']).sort_values(by = 'VIF', ascending = False)

        return vif_df
    
    @staticmethod
    def GetCorrelatedVars(corrMat, param, threshold):
        '''
        This method get the correlation matrix of the dataset. The correlation matrix is used to determine the correlation between the different features of the dataset.

        Inputs:
        corrMat: DataFrame with the correlation matrix of the dataset
        param: str, parameter to get the correlated variables
        threshold: float, threshold to get the correlated variables

        Returns:
        None
        '''
        #Get columns from correlation matrix
        aux = corrMat[param]

        # Fetch the values in the range: x>threshold AND x < -treshold
        aux = pd.concat([aux[aux > threshold], aux[aux < -threshold]]).abs()

        # Sort the values in descending order
        if type(aux) == pd.Series:
            aux = aux.sort_values(ascending=False)
        else:
            pass

        # Print the values
        return aux
    
    #MANU
    @staticmethod
    def get_corrMat(targets: list, correlation_matrix_type: str ="pearson", path: str = './data/2024-03-21/FS_split_dataset', only_features: bool = False):
        
        # Read the split datasets
        X_train, y_train, _, _ = Functions.ReadTrainTestSet(path)
        
        # Reset the index
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)

        # Extract only the targets from the training set
        if only_features:
            info = X_train
        else:
            target_data = y_train[targets]
            info = pd.concat([X_train, target_data], axis=1)

        corrMat = pd.DataFrame({"correlation_matrix_type": [correlation_matrix_type], "matrix_data": [info.corr(correlation_matrix_type)]})
        
        # Return the MI
        return corrMat
    
     #MANU
    @staticmethod
    def get_correlated_scores(targets: list, correlation_matrix: pd.DataFrame, threshold: float = 0.5, use_targets: bool = False):
       
        overall_correlated_df= pd.DataFrame()
        # Iterate over the targets
        for target in targets:

            # Extract the stats per target
            data_target = correlation_matrix.matrix_data.iloc[0].get(target)
            
            if not use_targets:
                data_target.drop(targets, inplace=True)
            else:
                data_target.drop(target, inplace=True)
                
            correlated_vars = pd.concat([data_target[data_target > threshold], data_target[data_target < -threshold]]).abs()
            
            if type(correlated_vars) == pd.Series:
                correlated_vars = correlated_vars.sort_values(ascending=False)
            else:
                pass
            
            # Initialize the feature data
            feature_data = pd.DataFrame()
            
            feature_data["Feature"] = list(correlated_vars.keys())
            feature_data["Correlated_score"] = list(correlated_vars.values)
            feature_data["Target"] = target
            feature_data["Threshold"] = threshold
            feature_data["Correlation_matrix"] = correlation_matrix.correlation_matrix_type.iloc[0]
            
            # Append the feature data per target to the overall feature data
            overall_correlated_df = pd.concat([overall_correlated_df, feature_data], axis = 0)
            
        return overall_correlated_df
    
    #Funciones DRAW
    
    @staticmethod
    def cm_to_inch(value):
        '''
        This method converts the value from centimeters to inches.

        Inputs:
        value: float, value to convert

        Returns:
        float, value converted to inches
        '''
        return value/2.54
    
    @staticmethod
    def GetTitleLabel(KQI, Replace = True):
        '''
        This method returns the title label in function of the KQI. The title label is used to label the figures.

        Inputs:
        KQI: str, KQI to get the title label

        Returns:
        output: str, title label
        '''
        #IF Replace is true, return a proper title for the figure
        if Replace:
            if(KQI == 'Reach1080p'):
                return 'Alcanza 1080p'
            elif(KQI == 'Reach1080p+'):
                return 'Alcanza HD'
            elif(KQI == 'Reach2K'):
                return 'Alcanza 2K'
            elif(KQI == 'Reach4K'):
                return 'Alcanza 4K'
            elif(KQI == '1080p'):
                return 'Porcentaje de la sesión a 1080p'
            elif(KQI == '2K'):
                return 'Porcentaje de la sesión a 2K'
            elif(KQI == '4K'):
                return 'Porcentaje de la sesión a 4K'
            elif(KQI == 'timeToReach1080p'):
                return 'Tiempo en alcanzar 1080p'
            elif(KQI == 'timeToReach1080p+'):
                return 'Tiempo en alcanzar HD'
            elif(KQI == 'timeToReach2K'):
                return 'Tiempo en alcanzar 2K'
            elif KQI == 'timeToReach4K':
                return 'Tiempo en alcanzar 4K'
            
            elif KQI == 'RSSI':
                return 'RSSI'
            elif KQI == 'RSSIDesv':
                return 'Desviación típica RSSI'
            elif KQI == 'BytesSent':
                return 'Throughput enviado por AP (Bytes)'
            elif KQI == 'BytesSentDesv':
                return 'Desviación típica Throughput enviado por AP (Bytes)'
            elif KQI == 'BytesReceived':
                return 'Throughput recibido por AP(Bytes)'
            elif KQI == 'BytesReceivedDesv':
                return 'Desviación típica throughput recibido por AP (Bytes)'
            elif KQI == 'PacketsSent':
                return 'Paquetes enviados por AP'
            elif KQI == 'PacketsSentDesv':
                return 'Desviación típica Paquetes enviados por AP'
            elif KQI == 'PacketsReceived':
                return 'Paquetes recibidos por AP'
            elif KQI == 'PacketsReceivedDesv':
                return 'Desviación típica Paquetes recibidos por AP'
            elif KQI == 'ErrorsSent':
                return 'Errores enviados por AP'
            elif KQI == 'ErrorsSentDesv':
                return 'Desviación tipica Errores enviados por AP'
            elif KQI == 'nAliens':
                return 'Número de aliens'
            elif KQI == 'nAliensDesv':
                return 'Desviación tipica Número de aliens'
            elif KQI == 'RSSI_channels':
                return 'RSSI interferente'
            elif KQI == 'RSSI_channelsDesv':
                return 'Desviación tipica RSSI interferente'
            else:
                return 'KQI not registered'
    @staticmethod
    def ReplaceLegendLabel(headers, Replace = False):
        '''
        This method replaces the labels of the legend in function of the headers. Updates the label of Machibe Learning models to simpler versions.

        Inputs:
        headers: list, list of headers
        Replace: bool, if True, replace the labels

        Returns:
        output: list, list of headers
        '''
        
        #If Replace is True, modify the headers
        if Replace:
            # Determine if 'RandomForest' or 'Ridge' labels exist
            # Then replace them by RFR and RR
            if (headers.count('RandomForest') + headers.count('Ridge') == 2):
                headers[headers.index('RandomForest')] = 'RF'
                headers[headers.index('Ridge')] = 'RR'

            elif(headers.count('RandomForest') == 1):
                headers[headers.index('RandomForest')] = 'RF'

            elif(headers.count('Ridge') == 1):
                headers[headers.index('Ridge')] = 'RR'
                
            return headers
        #If not, keep the same headers
        else:
            return headers
        
    @staticmethod
    def GetXlabel(metric):
        '''
        This method returns the Xlabel in function of the metric.

        Inputs:
        metric: str, metric to get the Xlabel

        Returns:
        output: str, Xlabel
        '''
        
        #Get the Xlabel in function of the metric
        if(metric == 'MAEP'):
            return 'MAE %' #MAE %
        elif (metric == 'pTime'):
            return 'Tiempo de Predicción (ms)'
        elif (metric == 'TrainingTime') :
            return 'Seconds (s)'
        elif metric == 'F1_weighted':
            return 'F1 weighted score (0-1)'

    @staticmethod
    def GetYLabel(options):
        '''
        This method returns the Ylabel in function of the options.

        Inputs:
        options: list, list of options

        Returns:
        output: list, list of Ylabels
        '''
        output = []
        # Get the Ylabel in function of the options
        for value in options:
            aux = value.split('_')[0]
            if aux == 'FS1':
                aux = 'FS'
            output.append(aux)
            
        # Return the values
        return output
    
    @staticmethod
    def FixDataScale(data, metric):
        '''
        This method fixes the data scale in function of the metric. (Percentaje metrics should be multiplied by 100)

        Inputs:
        data: DataFrame with the features of the dataset
        metric: str, metric to fix the scale

        Returns:
        data: DataFrame with the fixed scale
        '''
        #Fix the data scale in function of the metric. (Percentaje metrics should be multiplied by 100)
        if(metric == 'MAEP' or metric == 'RMSEP'):
            return data * 100
        elif(metric == 'pTime'):
            return data * 1000
        else:
            return data
        
    @staticmethod
    def save_figure(figure: Figure, name: str, path: str = './figures', format: str = 'pdf', additionalInfo: str = ''):
        '''
        This method saves a figure in a file in format pdf by default. Options are: png, pdf, svg, jpg, jpeg, tif, tiff, eps, raw, rgba, pgf, ps, and svgz.

        Inputs:
        figure: figure, figure to save
        name: str, name of the figure
        path: str, path to the directory where the figure will be saved

        Returns:
        None
        '''

        # Check if the path exists
        if Functions.CheckDirectory(path, msg=False):
            # If path is given, save the figure in the path provided, and not create a new directory with the today's date
            if path != './figures':
                today = ''
            else:
                # Create a new directory for the figures with the today's date inside the data directory
                today = pd.to_datetime('today').strftime('%Y-%m-%d')
                Functions.CheckDirectory(f'{path}/{today}', msg=False)

        # Define the pathname
        if additionalInfo != '':
            fileName = f'{path}/{today}/{additionalInfo}_{name}.{format}'
        else:
            fileName = f'{path}/{today}/{name}.{format}'

        # Save the figure
        figure.savefig(fileName, bbox_inches = 'tight')

        # Print confirmation
        print(colored(f'Figure saved as: {fileName}\n', 'yellow', attrs=['bold']))
        
    @staticmethod
    def plotBars(scores: pd.DataFrame, targets: list, metric: str = 'MAEP', orient: str = 'v', xlabel: str = 'Feature engineering mode', 
                 colormap: str = 'tab10', sns_style: str = 'whitegrid', context: str = 'poster', replaceTitles: bool = True, 
                 save_figure: bool = False, save_json: bool = False, additionalInfoName: str = '', path: str = './figures'):
        '''
        This method creates a bar graph with the scores of the models. The bar graph is created in function of the targets and the metric.

        Inputs:
        scores: DataFrame with the scores of the models
        targets: list, list of targets
        metric: str, metric to plot ('pTime' or Any if available)
        orient: str, orientation of the graph ('v' or 'h')
        xlabel: str, xlabel of the graph. Default: 'Feature engineering mode'
        colormap: str, colormap of the graph. Default: 'tab10'
        sns_style: str, style of the graph. Default: 'whitegrid'
        context: str, context of the graph. Default: 'poster'
        
        Returns:
        None
        '''

        sns.set_theme(style=sns_style) # type: ignore
        sns.set_context(context, font_scale=1.1) # type: ignore

        metricName = metric
        best_models = []
        for target in  targets:
            # Get a subset of the data
            data = scores[scores.target == target]

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(Functions.cm_to_inch(30), Functions.cm_to_inch(25)))

            # Get the metric label and best values for pTime
            if metric == 'pTime':
                metriclabel = Functions.GetXlabel(metric)

                # Get the least value of pTime
                best_algorithm = data.loc[data.pTime.idxmin()].algorithm
                best_value = data.loc[data.pTime.idxmin()].pTime
                best_feat_mode = data.loc[data.pTime.idxmin()].feat_mode
            
            # Get the metric label and best values for G Score
            elif metric == 'PET_score':
                metriclabel = 'PET score'

                # Get the best value of G Scores (minimum value)
                best_algorithm = data.loc[data.PET_score.idxmin()].algorithm
                best_value = data.loc[data.PET_score.idxmin()].PET_score
                best_feat_mode = data.loc[data.PET_score.idxmin()].feat_mode

            elif metric == 'comp_PET_score':
                metriclabel = 'Complementary PET score'

                # Get the best value of G Scores (maximum value)
                best_algorithm = data.loc[data.comp_PET_score.idxmax()].algorithm
                best_value = data.loc[data.comp_PET_score.idxmax()].comp_PET_score
                best_feat_mode = data.loc[data.comp_PET_score.idxmax()].feat_mode

            # Get the metric label and best values for the other metrics
            else:
                metriclabel = Functions.GetXlabel(data.metric.iloc[0])
                # Overwrite the metric label to fit any metric labelled as 'value'
                metric = 'value'

                # If ml_mode is regression, the lower the value the better
                if data.ml_mode.iloc[0] == 'regression':
                    best_algorithm = data.loc[data.value.idxmin()].algorithm
                    best_value = data.loc[data.value.idxmin()].value
                    best_feat_mode = data.loc[data.value.idxmin()].feat_mode

                else:
                    best_algorithm = data.loc[data.value.idxmax()].algorithm
                    best_value = data.loc[data.value.idxmax()].value
                    best_feat_mode = data.loc[data.value.idxmax()].feat_mode
            

            # Plot the data
            if orient == 'v':
                sns.barplot(data=data, y=metric, x = 'feat_mode', hue='algorithm', ax=ax, width=0.5, palette=colormap, orient='v')
            elif orient == 'h':
                sns.barplot(data=data, x=metric, y = 'feat_mode', hue='algorithm', ax=ax, width=0.5, palette=colormap, orient='h')

            # Highlight the best algorithm and display it in the graph
            # Draw a line at the best value
            if orient == 'v':
                ax.axhline(best_value, color='red', linestyle='--')
                # Edit the visible information
                ax.set_ylabel(metriclabel)
                ax.set_xlabel(xlabel)
            
            elif orient == 'h':
                ax.axvline(best_value, color='red', linestyle='--')
                # Edit the visible information
                ax.set_xlabel(metriclabel)
                ax.set_ylabel(xlabel)

            # Set title in black color and bold
            if replaceTitles:
                ax.set_title(Functions.GetTitleLabel(data.target.iloc[0], Replace=True))
            else:
                ax.set_title(data.target.iloc[0])
            ax.title.set_color('black')
            ax.title.set_fontweight('bold')
            ax.legend(reverse=False, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title = 'Algoritmo')
                
            # Show best value in the graph on the top left side
            ax.text(x=0.0, y=-0.2, s=f'* Linea roja discontinua: Mejor valor = {np.round(best_value, 2)} con {best_algorithm} y {best_feat_mode}', fontsize = 'small', color='black', fontstyle='italic', transform=ax.transAxes)

            
            best_models.append({
                "metric": metriclabel,
                "target": target,
                "feat_mode": best_feat_mode,
                "algorithm": best_algorithm,
                "value": np.round(best_value, 3)
            })
            
            # Save the figure
            if save_figure:
                Functions.save_figure(figure=fig, name=f'{metricName}_{target}', format='pdf', additionalInfo=additionalInfoName, path=path)

        if save_json:
            Functions.SaveJson(data=best_models, name=f'{path}/best_models_{metric}')
            
        return pd.DataFrame(best_models)

    @staticmethod
    def draw_cat_plot(feature_scores: pd.DataFrame, targets: list):
        '''
        This method draws a categorical plot with the feature scores of the models. The plot is created in function of the algorithms and the features.

        Inputs:
        feature_scores: DataFrame with the feature scores of the models

        Returns:
        None
        '''
        for target in  targets:
            # Extract the feature scores for the especified target
            data = feature_scores[feature_scores.target == target]
            print(data)
            # Draw a categorical plot per each algorithm
            plot = sns.catplot(data=data, x='Feature', y='Score', col='algorithm', kind='bar', hue='Feature', height=Functions.cm_to_inch(20), aspect=1, native_scale=True, legend='full', legend_out=True)

            # Rotate x-axis labels
            for ax in plot.axes.flat:
                for label in ax.get_xticklabels():
                    label.set_rotation(90)

                # Draw a line depicting the score
                title = ax.get_title().split('= ')[-1]

                # Create a twinx axis
                ax2 = ax.twinx()
                sns.lineplot(data=data[data.algorithm == title], x='Feature', y='model_score', color='red', sort=True, ax = ax2, legend=False)

                # Configure axes to be configured visually equal than the bar plot
                ax2.set_ylim((0, 1))

    @staticmethod
    def draw_heat_map(MI_df: pd.DataFrame, height_cm: float = 30, width_cm: float = 30, colormap: str = 'cividis', annot: bool = False, n_digits: int = 2, title: str = 'Mutual Information Matrix', 
                      xlabel_rotation: int = 45, cbar_label: str = 'Information (nat)', cbar_rotation: int = 270, cbar_pad: int = 30, save_figure: bool = False, additionalInfoName: str = '', 
                      path: str = './figures', draw_triangle: bool = False):
        '''
        This method draws a heatmap with the feature scores of the models. The heatmap is created in function of the algorithms and the features.

        Inputs:
        MI_df: DataFrame with the Mutual Information matrix
        height_cm: float, height of the figure in centimeters
        width_cm: float, width of the figure in centimeters
        colormap: str, colormap of the heatmap
        annot: bool, if True, show the values in the heatmap
        n_digits: int, number of digits to show in the heatmap
        title: str, title of the heatmap
        xlabel_rotation: int, rotation of the x-axis labels
        cbar_label: str, label of the colorbar
        cbar_rotation: int, rotation of the colorbar label
        cbar_pad: int, padding of the colorbar label
        save_figure: bool, if True, save the figure
        additionalInfoName: str, additional information to add to the name of the figure
        path: str, path to the directory where the figure will be saved
        draw_triangle: bool, if True, draw only the lower triangle of the heatmap

        Returns:
        None
        '''
        
        # If draw_triangle is True, draw only the lower triangle of the heatmap
        if draw_triangle:
            mask = np.triu(np.ones_like(MI_df, dtype=bool))
            MI_df = MI_df.mask(mask)
        else:
            mask = np.zeros_like(MI_df, dtype=bool)

        # Draw the heatmap
        fig, ax = plt.subplots(figsize=(Functions.cm_to_inch(width_cm), Functions.cm_to_inch(height_cm)))
        sns.heatmap(MI_df, annot=annot, fmt=f'.{n_digits}f', cmap=colormap, ax=ax, mask=mask, cbar_kws={'orientation': 'vertical'})

        # Set the title
        ax.title.set_text(title)
        ax.title.set_fontweight('bold')

        # Set label to the heatmap
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xlabel_rotation, horizontalalignment='right')

        # Set label to the colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar_label, rotation=cbar_rotation, labelpad=cbar_pad)

        # Save the figure
        if save_figure:
            Functions.save_figure(figure=fig, name=f'MI_heatmap', format='pdf', additionalInfo=additionalInfoName, path=path)

    @staticmethod
    def draw_NMI_heat_map(MI_df: pd.DataFrame, height_cm: float = 30, width_cm: float = 30, colormap: str = 'cividis', annot: bool = False, n_digits: int = 2, title: str = 'Mutual Information Matrix', 
                      xlabel_rotation: int = 45, cbar_label: str = 'Information (nat)', cbar_rotation: int = 270, cbar_pad: int = 30, save_figure: bool = False, additionalInfoName: str = '', 
                      path: str = './figures', draw_triangle: bool = False):
        '''
        This method draws a heatmap with the feature scores of the models. The heatmap is created in function of the algorithms and the features.

        Inputs:
        MI_df: DataFrame with the Mutual Information matrix
        height_cm: float, height of the figure in centimeters
        width_cm: float, width of the figure in centimeters
        colormap: str, colormap of the heatmap
        annot: bool, if True, show the values in the heatmap
        n_digits: int, number of digits to show in the heatmap
        title: str, title of the heatmap
        xlabel_rotation: int, rotation of the x-axis labels
        cbar_label: str, label of the colorbar
        cbar_rotation: int, rotation of the colorbar label
        cbar_pad: int, padding of the colorbar label
        save_figure: bool, if True, save the figure
        additionalInfoName: str, additional information to add to the name of the figure
        path: str, path to the directory where the figure will be saved
        draw_triangle: bool, if True, draw only the lower triangle of the heatmap

        Returns:
        None
        '''
        
        # If draw_triangle is True, draw only the lower triangle of the heatmap
        if draw_triangle:
            mask = np.triu(np.ones_like(MI_df, dtype=bool))
            MI_df = MI_df.mask(mask)
        else:
            mask = np.zeros_like(MI_df, dtype=bool)

        # Draw the heatmap
        fig, ax = plt.subplots(figsize=(Functions.cm_to_inch(width_cm), Functions.cm_to_inch(height_cm)))
        sns.heatmap(MI_df, annot=annot, fmt=f'.{n_digits}f', cmap=colormap, ax=ax, mask=mask, cbar_kws={'orientation': 'vertical'}, vmin=0, vmax=1)

        # Set the title
        ax.title.set_text(title)
        ax.title.set_fontweight('bold')

        # Set label to the heatmap
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xlabel_rotation, horizontalalignment='right')

        # Set label to the colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar_label, rotation=cbar_rotation, labelpad=cbar_pad)

        # Save the figure
        if save_figure:
            Functions.save_figure(figure=fig, name=f'MI_heatmap', format='pdf', additionalInfo=additionalInfoName, path=path)
            
    @staticmethod
    def draw_mi_perf_figure(targets: list, feature_scores: pd.DataFrame, MI_df: pd.DataFrame, xlabel_rotation: int = 90, verbose: bool = False, save_fig: bool = False, name: str = 'MIvsPerf', path: str = './figures', additionalInfoName: str = ''):
        '''
        This method draws a figure with the Mutual Information of the features and the performance of the models. The figure is created in function of the targets and the feature scores.

        Inputs:
        targets: list, list of targets
        feature_scores: DataFrame with the feature scores of the models
        MI_df: DataFrame with the Mutual Information of the input features with the targets
        xlabel_rotation: int, rotation of the x-axis labels
        verbose: bool, if True, print the scale properties
        save_fig: bool, if True, save the figure
        name: str, name of the figure
        path: str, path to the directory where the figure will be saved
        additionalInfoName: str, additional information to add to the name of the figure

        Returns:
        None
        '''

        # Iterate over the targets
        for target in targets:
            # Get the feature scores for the target
            data = feature_scores[feature_scores.target == target]

            # Extract the MI of the target
            MI_target = (MI_df.loc[target].drop(targets)).sort_values(ascending=False) # type: ignore

            # Create a figure
            fig, ax = plt.subplots(figsize=(Functions.cm_to_inch(30), Functions.cm_to_inch(30)))

            # Draw the original MI bars
            sns.barplot(x=MI_target.index, y=MI_target.values, hue=MI_target.index, palette='bright', ax=ax, 
                        alpha=0.5, width=0.75, facecolor=(1, 1, 1, 0), edgecolor='0.0', linestyle='--')

            # Plot the bars per feature
            plot = sns.barplot(data=data, x='Feature', y='Score', hue='algorithm', ax=ax, palette='tab10', 
                            saturation=1.0, alpha=0.75, width=0.5)
            plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
            plot.legend(reverse=False, bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0, title = 'Algoritmo')

            # Set the title
            ax.set_title(Functions.GetTitleLabel(target))
            ax.title.set_fontweight('bold')

            # Set the y1-axis label
            ax.set_ylabel('Información Mutua (nat)')

            # Set the x-axis label
            ax.set_xlabel('Característica')

            # Set the x-axis ticks rotation
            ax.set_xticklabels(data.Feature, rotation=xlabel_rotation, ha='right')

            ax2 = ax.twinx()
            # Draw a line depicting the score for every algorithm
            plot2 = sns.lineplot(data=data, x='Feature', y='model_score', hue='algorithm', sort=True, ax = ax2)

            # Set off the grid
            ax2.grid(False)

            # Get the maximum value of the y-axis
            y_max = ax.get_ylim()[1]

            # Get the maximum value of yticklabels
            y_ticks = ax.get_yticks()
            y_maxtick = y_ticks[-1]

            # Count the number of ytickslabel in the first y-axis
            n_ticks = len(ax.get_yticklabels())

            # Get the maximum value of model_score
            y_max_value2 = data.model_score.max()

            # Find the closest multiple of yticklabel from ax1 that is less than y_max_value2
            proportion = y_max_value2/y_max

            # If the proportion is greater than or equal to 1
            if proportion >= 1:
                y_tick_factor = np.round(y_max_value2/y_max)

                # Find the maximum value of yticklabel that is less than y_max_value2
                y_maxtick2 = y_maxtick * y_tick_factor

                # Determine the power of ten to be used
                power = np.abs(np.floor(np.log10(y_max_value2))) + 1
                units_left = n_ticks - ((y_maxtick2 * np.power(10, power)) % n_ticks)

                # Determine the max value for yticks in the second axis
                y_maxtick2 = y_maxtick2 + units_left/np.power(10, power)

                # Create a np.linspace from 0 to y_max_value2
                y_ticks = np.linspace(0, y_maxtick2, n_ticks + 1)

                # Replace the yticks of the second axis
                ax2.set_yticks(y_ticks)
                
                # Calculate ylim2 as function of the y_tick_factor
                y_max2 = (y_ticks[-2] * y_max) / y_maxtick
                y_add = 0
            # If the proportion is less than 1
            else:
                y_tick_factor = np.floor(y_max/y_max_value2)

                # Find the maximum value of yticklabel that is less than y_max_value2
                y_maxtick2 = (y_maxtick / y_tick_factor)
                
                # Determine the power of ten to be used
                power = np.abs(np.floor(np.log10(y_max_value2))) + 1
                #units_left = n_ticks - ((y_maxtick2 * np.power(10, power)) % n_ticks) 

                factor = np.ceil((y_max_value2 * np.power(10, power)) / (n_ticks - 1)) + 1
                # Determine the max value for yticks in the second axis
                #y_maxtick2 = y_maxtick2 + units_left/np.power(10, power)
                y_maxtick2 = (factor * (n_ticks -1)) / np.power(10, power)

                # Create a np.linspace from 0 to y_max_value2
                #y_ticks2 = np.linspace(0, y_maxtick2, n_ticks + 1)
                y_ticks2 = np.linspace(0, y_maxtick2, n_ticks)
                # Replace the yticks of the second axis
                ax2.set_yticks(y_ticks2)
                
                # Calculate ylim2 as function of the y_tick_factor
                #y_max2 = (y_ticks[n_ticks] * y_max) / y_maxtick
                y_add = (y_max - y_ticks[n_ticks - 2]) / y_ticks[n_ticks - 2]

                y_max2 = y_ticks2[n_ticks - 2] + y_ticks2[n_ticks - 2]*y_add

            # Set the ylim for the second axis
            ax2.set_ylim(0, y_max2)

            if verbose:
                # Scale properties
                print(f'********************* {target} *********************')
                print(f'y1 ticks =  {ax.get_yticks()}')
                print(f'y1 axis max = {y_max}')
                print(f'y1 max tick = {y_maxtick}')
                print(f'y1 n_ticks = {n_ticks}')
                print(f'y2 max value = {y_max_value2}')
                print(f'proportion = {proportion}')
                print(f'Adjust after last tick = {y_add}')
                print(f'y2 axis max = {y_max2}')
                print(f'y2 ticks = {ax2.get_yticks()}')
                print(f'y2 max tick ={y_maxtick2}')

            # Get the index of the point with the least model_score
            # If the target ml_mode is Regression, look for the minimum model_score
            if data.ml_mode.unique() == 'regression':
                best_score = data.groupby('algorithm').model_score.idxmin().to_frame()
            else:
                best_score = data.groupby('algorithm').model_score.idxmax().to_frame()


            # Set the legend
            plot2.legend(reverse=False, bbox_to_anchor=(1.2, 0.3), loc='upper left', borderaxespad=0, title = 'Perf.')

            # Draw a marker at the point with the least model_score in front of the line
            sns.scatterplot(data=data.loc[best_score.model_score], x='Feature', y='model_score', markers=True, hue='algorithm', 
                            hue_order=data.algorithm.unique(), ax=plot2, legend=False, style='algorithm', 
                            style_order=data.algorithm.unique(), s=200)
            
            if data.ml_mode.unique() == 'regression':
                ax2.set_ylabel('MAE %', rotation=270, labelpad=25)
                ax2.set_yticklabels(np.round(ax2.get_yticks() * 100, 2), rotation=0)
            else:
                ax2.set_ylabel('F1 score', rotation=270, labelpad=25)
                ax2.set_ylim(0.40,1.02)
            
            # Save the figure
            if save_fig:
                Functions.save_figure(fig, name=f'{name}_{target}', path=path, additionalInfo=additionalInfoName)
                
                
    #FUNCIONES DE BEST MODELS

    @staticmethod
    def get_best_model_parameters(data: pd.DataFrame):
        '''
        This method gets the best model statistics from the data. The best model statistics are the statistics of the best model in the data.

        Inputs:
        data: DataFrame with the statistics of the models

        Returns:
        best_model_stats: dict, dictionary with the statistics of the best model
        targets_in_data: list, list of targets in the data
        '''
        
        #  Initialize the counter
        counter = 0

        # Get the feature modes present in the data
        targets_in_data = list(data.keys())

        # Initialize the best_params dictionary
        best_params = {}
        
        # Iterate over the targets in the passed data
        for target in targets_in_data:
            # Get the data for the target
            data_target = data.get(target)

            # Get the feature modes present in the data
            feature_modes = list(data_target.keys())

            # Iterate over the feature modes
            for mode in feature_modes:

                # Get the data for the feature mode
                data_mode = data_target.loc[mode]

                # Get the algorithms present in the data
                algorithms = list(data_mode.keys())

                # Iterate over the algorithms
                for algorithm in algorithms:
                    
                    # Append the metrics value and ml_model to the best_params dictionary
                    ml_mode = data_mode[algorithm].get('ml_mode')
                    metrics = data_mode[algorithm].get('scores')

                    # Get the metric according the ml_mode
                    if ml_mode == 'regression':
                        value = metrics.get('MAEP')
                        metric = 'MAEP'
                    elif ml_mode == 'classification':
                        value = metrics.get('F1_weighted')
                        metric = 'F1_weighted'
                    # Get the model prediction time
                    pTime = metrics.get('pTime')

                    # Get the best model parameters
                    model_params = {}
                    model_params['model_parameters'] = data_mode[algorithm].get('model')

                    # Append values
                    model_params['value'] = value
                    model_params['metric'] = metric
                    model_params['pTime'] = pTime
                    model_params['ml_mode'] = ml_mode
                    model_params['algorithm'] = algorithm
                    model_params['target'] = target
                    model_params['feature_mode'] = mode

                    # Save the best_params dictionary
                    best_params[counter] = model_params
                    
                    # Increase the counter
                    counter += 1

        # Convert the best_params dictionary to a DataFrame
        best_params = pd.DataFrame.from_dict(best_params, orient='index')

        # Return the best_params dictionary
        return best_params, targets_in_data

    @staticmethod
    def get_best_model_overall(best_params: pd.DataFrame, use_PET: bool = True):
        '''
        This method gets the best model parameters overall from the data. 
        The best model parameters overall are the parameters with the least PET_score (If enabled) or 
        the minimum error (MAEP by default) or maximum  classification quality (F1 score weighted by default).

        Inputs:
        best_params: DataFrame with the statistics of the models
        use_PET: bool, if True, use the PET score to get the best model

        Returns:
        best_options: DataFrame with the best model parameters overall
        '''
        
        # Initialize the counter and the best option dataframe
        counter = 0
        best_options = pd.DataFrame()

        # Find the unique targets and algorithms
        targets = best_params.target.unique().tolist()
        algorithms = best_params.algorithm.unique().tolist()

        # Iterate over the targets and algorithms
        for target in targets:
            for algorithm in algorithms:
                # Search the subset that has the same algorithm and target
                subset_index = best_params[best_params.target == target].index.intersection(best_params[best_params.algorithm == algorithm].index)

                # Get the best paramters overall in the subset
                subset = best_params.loc[subset_index]

                # If the subset could be found:
                if len(subset) > 0:
                    # If use_PET is True, the best option is the minimum value
                    if use_PET:
                        best_option = subset.loc[subset.PET_score.idxmin()].to_frame().T
                    else:
                        # If ml_mode is regression, the best option is the minimum MAEP, elif it is classification, the best option is the maximum f1_score
                        if subset.ml_mode.iloc[0] == 'regression':
                            best_option = subset.loc[subset.metric.idxmin()].to_frame().T
                        else:
                            best_option = subset.loc[subset.metric.idxmax()].to_frame().T
                        
                    # Append the best option per target and algorithm
                    best_options = pd.concat([best_options, best_option], axis=0)

                    # Increase the counter
                    counter += 1

        # Reset the index
        best_options.reset_index(inplace=True, drop=True) 

        # Return the best options
        return best_options
    
    @staticmethod
    def get_best_model_params_in_path(path: str, save_json: bool = False, path_save: str = './stats_models', best_model_overall: bool = False, use_PET: bool = True):
        '''
        This method gets the best model parameters from the files in a directory. The best model parameters are the parameters of the best model in the files.

        Inputs:
        path: str, path to the directory where the files are located

        Returns:
        best_params_dict: dict, dictionary with the best model parameters
        best_params: DataFrame with the best model parameters
        best_params_overall: DataFrame with the best model parameters overall
        '''
        
        # Get the filenames in a directory
        files = Functions.get_filenames_in_directory(path)

        # Initialize the dictionary to store the best model parameters
        best_params = pd.DataFrame()

        for file in files:
            # Discard any file that have 'overall' in the name
            if 'overall' in file:
                pass
            elif 'summary' in file:
                pass
            else:
                # Read the data from the file
                data = pd.read_json(path + file)

                # Get the best model parameters
                best_params_target, targets_in_data = Functions.get_best_model_parameters(data)

                # Concatenate the string items in the list to a single string
                key = ''.join([str(elem) for elem in targets_in_data])

                # Append the best model parameters to the dictionary
                best_params_target['used_keys'] = key

                # Concat the dataframe
                best_params = pd.concat([best_params, best_params_target], axis=0).reset_index(drop=True)

        # Convert the best_params DataFrame to a dictionary
        best_params_dict = pd.DataFrame.to_dict(best_params, orient='index')

        # Get the best model overall
        if best_model_overall:
            Functions.estimate_pet_score(scores=best_params)
            best_params_overall = Functions.get_best_model_overall(best_params, use_PET=use_PET)

            # If save flag is active, save the dictionary in a json file
            if save_json:
                # Save the dictionary in a json file
                Functions.SaveJson(data=best_params_dict, name=f'{path_save}summary_best_params')
                Functions.SaveJson(data=best_params_overall.to_dict(), name=f'{path_save}summary_best_params_overall')
        else:
            best_params_overall = pd.DataFrame()

            # If save flag is active, save the dictionary in a json file
            if save_json:
                # Save the dictionary in a json file
                Functions.SaveJson(data=best_params_dict, name=f'{path_save}summary_best_params')
            
        # Return the information
        return best_params_dict, best_params, best_params_overall
    
    
    #FUNCIONES DE MAPEO
        
    @staticmethod
    def map_feature_eng_mode(feat_eng_mode: str|list):
        '''
        This methods maps the feature engineering mode to a short version. The short version is used to name the models and the files.

        Inputs:
        feat_eng_mode: str, feature engineering mode to map

        Returns:
        str, short version of the feature engineering mode
        '''

        # If feat_eng_mode is 'FS1', return 'FS'
        if feat_eng_mode == 'Feature_selection':
            return 'FS'
        # If feat_eng_mode is 'PCA', return 'PCA'
        elif feat_eng_mode == 'Feature_extraction':
            return 'FE'
        # If feat_eng_mode is 'No_FE', return 'No_FE'
        elif feat_eng_mode == 'No_FE':
            return 'No_FE'
        # If string is empty, return 'No_FE'
        else:
            return 'No_FE'
        
    @staticmethod
    def map_acronym_to_feature_eng_mode(feat_label: str):
        '''
        This methods maps the acronym of the feature engineering mode to the full name of the feature engineering mode.

        Inputs:
        feat_label: str, acronym of the feature engineering mode

        Returns:
        str, full name of the feature engineering mode
        '''

        # If the input is string, map directly, if it is a list, map each element
        if isinstance(feat_label, list):
            aux = []
            for label in feat_label:
                if label == 'FS':
                    label = 'Feature_selection'
                elif label == 'FE':
                    label = 'Feature_extraction'
                elif label == 'No_FE':
                    label = 'No_FE'
                else:
                    label = 'No_FE'
                
                # Append the label to the list
                aux.append(label)
            return aux
            
        else:
            if feat_label == 'FS':
                feat_label = 'Feature_selection'
            elif feat_label == 'FE':
                feat_label = 'Feature_extraction'
            elif feat_label == 'No_FE':
                feat_label = 'No_FE'    
            else:
                feat_label = 'No_FE'
            # Return the label
            return feat_label

    @staticmethod
    def MapResolution(dataSet, totalSamp, nSamp):
        '''
        This method maps the resolution of the video to an integer number. The integer number is used to represent the resolution of the video.   

        Inputs: 
        dataSet: DataFrame with the features of the dataset
        totalSamp: int, total number of samples in the dataset
        nSamp: int, number of samples used to map the resolution

        Returns:
        None
        '''

        # Display a warning message
        Functions.deprecated()

        # Extract videoWidth column to map resolution to integer number
        res = dataSet.videoWidth
        resInd = []
        resProfile = []

        for i in range(0, totalSamp, nSamp):

            for j in range(0, nSamp):

                # Map the resolution value to an integer
                if(res[i + j] == 0):
                    resInd.append(0)
                elif(res[i + j] == 720):
                    resInd.append(1)
                elif(res[i + j] == 1080):
                    resInd.append(2)
                elif(res[i + j] == 1440):
                    resInd.append(3)
                elif(res[i + j] == 2160):
                    resInd.append(4)
                else:
                    resInd.append(5)
    
                # Generate a profile, if resolution increases, so does the counter, if resolution decreases, so does the counter
                if(j == 0):
                    resProfile.append(0)
                    auxRes = 0
                else:
                    if(resInd[i + j] > resInd[i + j -1]):
                        auxRes += resInd[i + j] - resInd[i + j -1]
                        resProfile.append(auxRes)
                    elif(resInd[i + j] < resInd[i + j -1]):
                        auxRes -= resInd[i + j - 1] - resInd[i + j]
                        resProfile.append(auxRes)
                    else:
                        resProfile.append(auxRes)

        # Insert the new columns into the dataframe
        indAux = dataSet.columns.get_loc('resolutionSwitches') + 1
        dataSet.insert(indAux, 'resolution', resInd)   
        dataSet.insert(indAux + 1, 'resProfile', resProfile) 
        
    @staticmethod    
    def MapResolution2(dataSet, totalSamp, nSamp):
        '''
        This method maps the resolution of the video to an integer number. The integer number is used to represent the resolution of the video.

        Inputs:
        dataSet: DataFrame with the features of the dataset
        totalSamp: int, total number of samples in the dataset
        nSamp: int, number of samples used to map the resolution

        Returns:
        None
        '''

        # Extract videoWidth column to map resolution to integer number
        res = dataSet.videoWidth
        resInd = []
        resProfile = []

        for i in range(0, totalSamp, nSamp):

            for j in range(0, nSamp):

                #Map the resolution value to an integer
                if(res.iloc[i + j] == 0):
                    resInd.append(0)
                elif(res.iloc[i + j] == 720):
                    resInd.append(1)
                elif(res.iloc[i + j] == 1080):
                    resInd.append(2)
                elif(res.iloc[i + j] == 1440):
                    resInd.append(3)
                elif(res.iloc[i + j] == 2160):
                    resInd.append(4)
                else:
                    resInd.append(5)

                # Generate a profile, if resolution increases, so does the counter, if resolution decreases, so does the counter
                if(j == 0):
                    resProfile.append(0)
                    auxRes = 0
                else:
                    if(resInd[i + j] > resInd[i + j -1]):
                        auxRes += resInd[i + j] - resInd[i + j -1]
                        resProfile.append(auxRes)
                    elif(resInd[i + j] < resInd[i + j -1]):
                        auxRes -= resInd[i + j - 1] - resInd[i + j]
                        resProfile.append(auxRes)
                    else:
                        resProfile.append(auxRes)

        #Insert the new columns into the dataframe
        indAux = dataSet.columns.get_loc('resolutionSwitches') + 1
        dataSet.insert(indAux, 'resolution', resInd)   
        dataSet.insert(indAux + 1, 'resProfile', resProfile) 

    @staticmethod
    def map_ml_algorithm_labels(label: str):
        '''
        This method maps the machine learning algorithm label to a shorter version. The shorter version is used to name the models and the files.

        Inputs:
        label: str, label of the machine learning algorithm

        Returns:
        str, short version of the machine learning algorithm
        '''

        # Check if the label is RR
        if label == 'RR':
            return 'RC'
        # Check if the label is KNR
        elif label == 'KNR':
            return 'KNC'
        # Check if the label is MLP
        elif label == 'MLP':
            return 'MLPC'
        # If not present, return the label
        else:
            return label
        

    #FUNCIONES DE CARGA Y CONFIGURACION
    
    @staticmethod
    def configure_ml_experiment(configuration_dict: dict, feature_modes: list|str, algorithms: list|str, targets: list|str) -> tuple[list, list, list, list, list, dict]:
        '''
        This method configures the machine learning experiment. The configuration is used to define the feature engineering modes, the algorithms and the targets to train.

        Inputs:
        configuration_dict: dict, dictionary with the configuration of the model
        feature_modes: list|str, list of feature engineering modes
        algorithms: list|str, list of algorithms
        targets: list|str, list of targets

        Returns:
        KQIs: list, list of KQIs
        drop_features: list, list of features to drop
        feature_modes: list, list of feature engineering modes
        algorithms: list, list of algorithms
        targets: list, list of targets
        '''

        # Get the type of KQI (discrete or continuous)
        KQI_type = configuration_dict.get('KQI_types')
        if not isinstance(KQI_type, dict):
            KQI_type = {}

        # Get the KQIs list
        if isinstance(KQI_type, dict):
            KQIs = list(KQI_type.keys())
        else:
            KQIs = []
        print(f'KQI list: \n{KQIs}\n')

        # Get the features to drop
        drop_features = configuration_dict['Drop_features']
        print(f'Features to drop: \n{drop_features}\n')

        # If the feature_mode is a string, put it into list format
        if isinstance(feature_modes, str):
            feature_modes = [feature_modes]

        # If the algorithm is a string, put it into list format
        if isinstance(algorithms, str):
            algorithms = [algorithms]

        # If the target is a string, put it into list format
        if isinstance(targets, str):
            targets = [targets]


        print(f'To train:\n\tFeature engineering modes: {feature_modes}\n\tAlgorithms: {algorithms} \n\tTargets: {targets}\n')

        return KQIs, drop_features, feature_modes, algorithms, targets, KQI_type

    @staticmethod
    def load_configuration(filepath: str = './config'):
        '''
        This method loads the configuration of the model from a json file. The configuration defines the KQIs to read, the type of variable (continuous or discrete) and the features to drop.

        Inputs:
        path: str, path to the directory where the configuration file is located

        Returns:
        config: dict, dictionary with the configuration of the model
        '''

        # Read the file
        config = Functions.LoadJson(filepath)

        # Return the configuration
        return config

    @staticmethod
    def load_dataset(path: str = r'./data/datasetProm.json', targets: list = [], drop_features: list = []) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        This method loads the dataset from a json file. The dataset is loaded as a DataFrame. The KQIs are extracted from the dataset and the features especified in the config.json file are dropped.

        Inputs:
        path: str, path to the directory where the dataset file is located
        targets: list, list of KQIs to extract from the dataset

        Returns:
        X: DataFrame with the features of the dataset
        '''
        
        # Load the dataset
        df = pd.read_json(path)

        # Extract the KQIs from the dataset
        Y = df[targets]

        # Drop the KQIs from the features dataset
        X = df.drop(columns = targets)

        # Drop the features that are not useful predictors
        X.drop(columns = drop_features, inplace=True)

        # Return the dataset
        return X, Y

    @staticmethod
    def SaveModel(model, pathname):
        '''
        This method saves a model in a file in format joblib.

        Inputs:
        model: model, model to save
        pathname: str, path to the file to save

        Returns:
        None
        '''

        # Import the necessary packages
        from joblib import dump

        # Save the model in a file
        dump(model, pathname)

        # Print confirmation
        print(colored(f'Model saved as: {pathname}\n', 'yellow', attrs=['bold']))
    
    @staticmethod
    def LoadModel(filepath: str):
        '''
        This method loads a model from a file in format joblib.

        Inputs:
        filepath: str, path to the file to load

        Returns:
        model: model, model loaded from the file
        '''
        
        # Import the necessary packages
        from joblib import load

        # Load the model from the file
        model = load(filepath)

        # Return the model
        return model

    @staticmethod
    def SaveJson(data: dict, name: str):
        '''
        This method saves a dictionary in a json file.

        Inputs:
        data: dict, dictionary to save
        name: str, name of the file to save without the extension

        Returns:
        None
        '''

        # Save the dictionary in a json file
        fileName = f'{name}.json'

        with open(fileName, "w") as file:
            json.dump(data, file, indent=4)

        print(colored(f'JSON file have been saved in: {fileName}', 'green', attrs=['bold']))

    @staticmethod    
    def LoadJson(name: str):
        '''
        This method loads a json file and returns the dictionary of the file.

        Inputs:
        name: str, name of the file to load without the extension

        Returns:
        dictionary: dict, dictionary of the json file
        '''
        import json

        fileName = '{}.json'.format(name)
        file = open(fileName, "r")
        dictionary = json.load(file)

        return dictionary
    
    @staticmethod
    def save_stats(stats: dict, target: str, path: str = './stats_models', message: bool = False):
        '''
        This method saves the statistics of the model in a json file.

        Inputs:
        target: str, name of the target
        path: str, path to the directory where the stats will be saved

        Returns:
        None
        '''

        # Import the necessary packages
        import json

        # Check if the data path exists
        if Functions.CheckDirectory(path, msg=message):
            # Create a new directory for the split datasets with the today's date inside the data directory
            today = pd.to_datetime('today').strftime('%Y-%m-%d')
            Functions.CheckDirectory(f'{path}/{today}', msg=message)

        # Save the dictionary in a json file
        fileName = f'{path}/{today}/stats_{target}'
        Functions.SaveJson(stats, fileName)
    
    @staticmethod
    def save_ml_model(model_obj, name: str, target: str, ml_mode: str, feat_eng_mode: str, use_kqi: bool = False, format: str = 'joblib', path: str = './models'):
        '''
        This method saves a machine learning model in a file in format joblib. The model is saved in a directory that is named in function of the machine learning mode and feature engineering mode.

        Inputs:
        model_obj: model, model object to save
        name: str, name of the ml algorithm
        ml_mode: str, machine learning mode
        feat_eng_mode: str, feature engineering mode
        use_kqi: bool, if True, use the KQI to name the model
        path: str, path to the directory where the model will be saved

        Returns:
        None
        '''

        # Check if the path exists
        if Functions.CheckDirectory(path, msg=False):
            # Create a new directory for the models with the today's date inside the data directory
            today = pd.to_datetime('today').strftime('%Y-%m-%d')
            Functions.CheckDirectory(f'{path}/{today}', msg=False)

        # Define the pathname
        feat_eng_mode = Functions.map_feature_eng_mode(feat_eng_mode)

        # If use_kqi is True, use the KQI to name the model
        if use_kqi:
            fileName = f'{path}/{today}/{name}_{target}_{ml_mode}_{feat_eng_mode}_WKQI.{format}'
        else:
            fileName = f'{path}/{today}/{name}_{target}_{ml_mode}_{feat_eng_mode}_NoKQI.{format}'

        # Save the model
        Functions.SaveModel(model_obj, fileName)

    @staticmethod
    def load_ml_model(name: str, target: str, ml_mode: str, feat_eng_mode: str, use_kqi: bool = False, format: str = 'joblib', path: str = './models'):
        '''
        This method loads a machine learning model from a file in format joblib. The model is loaded from a directory that is named in function of the machine learning mode and feature engineering mode.

        Inputs:
        name: str, name of the ml algorithm
        ml_mode: str, machine learning mode
        feat_eng_mode: str, feature engineering mode
        use_kqi: bool, if True, use the KQI to name the model
        format: str, format of the file (default: joblib)
        path: str, path to the directory where the model is located

        Returns:
        model: model, model loaded from the file
        '''

        # Define the pathname
        feat_eng_mode = Functions.map_feature_eng_mode(feat_eng_mode)

        # If use_kqi is True, use the KQI to name the model 
        if use_kqi:
            fileName = f'{path}/{name}_{target}_{ml_mode}_{feat_eng_mode}_WKQI.{format}.{format}'
        else:
            fileName = f'{path}/{name}_{target}_{ml_mode}_{feat_eng_mode}_NoKQI.{format}.{format}'

        # Load the model
        model = Functions.LoadModel(fileName)

        # Return the model
        return model

    @staticmethod
    def get_filenames_in_directory(directory: str):
        '''
        This method gets the filenames in a directory.

        Inputs:
        directory: str, path to the directory

        Returns:
        list, list of filenames in the directory
        '''

        # Get the list of files in the directory
        list = os.listdir(directory)

        # Return the list
        return list

    @staticmethod
    def get_number_files_in_directory(directory):
        '''
        This method gets the number of files in a directory.

        Inputs:
        directory: str, path to the directory

        Returns:
        int, number of files in the directory
        '''

        # Get the list of files in the directory
        list = Functions.get_filenames_in_directory(directory)
        
        # Get the number of files in the directory
        number_files = len(list)

        # Return the number of files
        return number_files
    
    @staticmethod
    def deprecated():
        '''
        This method is a decorator that is used to mark functions as deprecated. The decorator will display a warning message when the function is used. 
        
        Inputs:
        None

        Returns:
        None
        '''
        warn("deprecated", DeprecationWarning)
