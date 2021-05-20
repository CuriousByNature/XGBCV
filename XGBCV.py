import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import itertools

class xgb_cv:
    ##############################################################################################################
    # A class used to run cross validation for XGBoost with early stopping rounds when using the scikit-learn API 
    
    # Initialization
    #----------------------------------------------------------
    # The class is initialized using the XGBoost model object in the scikit-learn API
    
    # Methods
    #----------------------------------------------------------
    # The only method of the class is run_cv, which performs the cross validation. The inputs are (with default values in square brackets):
    # run_cv inputs:
    # X                     - the training feature dataframe
    # Y                     - the training array of target values
    # folds                 - [5], number of folds in the cross-validation
    # early_stopping_rounds - [None], number of early stopping rounds
    # eval_metric           - [None], the evaluation metric(s) to be used for cross-validation. If a list is given, the cross-validation score 
    #                         is computed for each
    # shuffle               - [False], boolean to denote whether folds should be shuffled
    # seed                  - [0], the random_state seed for the KFold object of the cross-validation
    # stratified            - [False], boolean to denote whether folds should be stratified
    #----------------------------------------------------------
    
    # Attributes
    #----------------------------------------------------------
    # The class variables are as follows:
    # optimal_iter          - array of the optimal (determined by early stopping rounds) number of epochs to run XGBoost for each fold
    # optimal_scores        - dictionary of list of scores on each validation fold for each eval_metric at the optimal_iter
    # optimal_train_scores  - dictionary of list of scores on each training fold for each eval_metric at the optimal_iter
    # final_iter            - array of the final (determined by early stopping arounds) epoch number for XGBoost for each fold
    # final_scores          - dictionary of list of scores on each validation fold for each eval_metric at the final_iter
    # final_train_scores    - dictionary of list of scores on each training fold for each eval_metric at the final_iter
    # model                 - the XGBoost model object
    # folds                 - number of folds used in run_cv
    # stratified            - denotes the value of stratified used in run_cv
    ##############################################################################################################
    
    def __init__(self,model):
        self.optimal_iter = np.array([])
        self.optimal_scores = {}
        self.optimal_train_scores = {}
        self.final_iter = np.array([])
        self.final_scores = {}
        self.final_train_scores = {}
        self.model = model
        self.folds = 0
        self.stratified = False
        
    def run_cv(self, X, Y, folds = 5, early_stopping_rounds = None, eval_metric = None, shuffle = False, seed = 0, stratified = False):
        self.folds = folds
        self.stratified = stratified
        
        #compute the folds, be it stratified or not
        if stratified is True:
            kfold = StratifiedKFold(n_splits = folds, shuffle = shuffle, random_state = seed)
            splits = kfold.split(X, Y)
        else:
            kfold = KFold(n_splits = folds, shuffle = shuffle, random_state = seed)
            splits = kfold.split(X)
            
        #create a dictionary with keys corresponding to the different eval metrics
        if not isinstance(eval_metric,list):
            eval_metric = [eval_metric]
            
        for metric in eval_metric:
            self.optimal_scores[metric] = np.array([])
            self.final_scores[metric] = np.array([])
            self.optimal_train_scores[metric] = np.array([])
            self.final_train_scores[metric] = np.array([])
            
        #perform the cross-validation
        for train_index, test_index in splits:
            X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
            y_train, y_test = Y[train_index], Y[test_index]
            eval_set = [(X_train, y_train), (X_test, y_test)]
            
            #Keep in mind that early stopping is evaluated based on the last metric listed in eval_metric
            #The eval set is set to be the current validation fold:
            self.model.fit(X_train, y_train, early_stopping_rounds = early_stopping_rounds, eval_set = eval_set,
                          eval_metric = eval_metric, verbose = False)
            results = self.model.evals_result()
            
            #store the results
            ii = 0
            for metric in eval_metric:
                results_metric = results['validation_1'][metric]
                results_metric_train = results['validation_0'][metric]
                if ii == 0:
                    final_iter = len(results_metric)-1
                    self.final_iter = np.append(self.final_iter, final_iter)
                    best_iter = final_iter #if no early stopping is used, best iter is just the last iter
                    
                    if isinstance(early_stopping_rounds, int): #then early stopping has been specified
                        best_iter = self.model.best_iteration
                    
                    self.optimal_iter = np.append(self.optimal_iter, best_iter)
                    ii += 1
                    
                self.optimal_scores[metric] = np.append(self.optimal_scores[metric], results_metric[best_iter])
                self.final_scores[metric]   = np.append(self.final_scores[metric], results_metric[final_iter])
                self.optimal_train_scores[metric] = np.append(self.optimal_train_scores[metric], results_metric_train[best_iter])
                self.final_train_scores[metric]   = np.append(self.final_train_scores[metric], results_metric_train[final_iter])



class xgb_GridSearchCV:
    ##############################################################################################################
    # A class used to run Grid Search Cross-Validation for XGBoost with early stopping rounds when using the scikit-learn API.
    # This class uses the xgb_cv class.
    
    # Initialization
    #----------------------------------------------------------
    # The class is initialized using the XGBoost model object in the scikit-learn API
    
    
    # Methods
    #----------------------------------------------------------
    # The only method of the class is run_GridSearchCV, which performs a grid search based on the parameter grid provided and runs
    # a cross validation at each point in the grid to evaluate the model in order to select the optimal set of parameters.
    
    # run_GridSearchCV inputs:
    # X                     - the training feature dataframe
    # Y                     - the training array of target values
    # folds                 - [5], number of folds in the cross-validation
    # early_stopping_rounds - [None], number of early stopping rounds
    # eval_metric           - [None], the evaluation metric(s) to be used for cross-validation. If a list is given, 
    #                         grid search will base its search evaluation on the last metric listed 
    # shuffle               - [False], boolean to denote whether folds should be shuffled
    # seed                  - [0], the first random_state seed for the KFold object of the cross-validation. For each subsequent 
    #                       - point in the grid, the seed is incremented by 1.
    # stratified            - [False], boolean to denote whether folds should be stratified
    #----------------------------------------------------------
    
    # Attributes
    #----------------------------------------------------------
    # The class variables are as follows:
    # best_metric_value     - the best value (i.e. as evaluated on the average of the validation folds) of the evaluation metric
    # best_trees            - array of the best number of n_estimators (for each fold) as determined by early stopping, at the optimal grid point
    # best_params           - the set of parameters that produced the best evaluation score
    # best_model            - the XGBoost model object instantiated using the best_params
    ##############################################################################################################
    def __init__(self,model):
        self.best_metric_value = np.inf
        self.best_trees = []
        self.best_params = {}
        self.model = model
        self.best_model = model
        
    def run_GridSearchCV(self, X, Y, param_grid, folds = 5, early_stopping_rounds = None, eval_metric = None, shuffle = False, seed = 0, stratified = False):
        ii = 0
        seed_grid = seed
        
        param_dict = dict((el,0) for el in param_grid.keys())
        param_dict_list = list(param_grid.keys())
        
        iterlist = list(itertools.product(*param_grid.values()))
        for params in iterlist:
            ii += 1
            
            #set up the parameter dictionary
            jj = 0 
            for val in params:
                param_dict[param_dict_list[jj]] = val
                jj += 1
                
            self.model.set_params(**param_dict)
            xgbcv = xgb_cv(self.model)
            xgbcv.run_cv(X, Y, folds = folds, early_stopping_rounds = early_stopping_rounds, eval_metric = eval_metric, shuffle = shuffle, seed = seed_grid, stratified = stratified)
            
            
            if isinstance(eval_metric,list):
                #then grid search will base its search evaluation on the last metric listed
                search_metric = eval_metric[-1]
            else:
                search_metric = eval_metric
            
            best_metric = xgbcv.optimal_scores[search_metric].mean()
            optimal_num_trees = xgbcv.optimal_iter+1
            seed_grid += 1
              
            if best_metric < self.best_metric_value:
                self.best_metric_value = best_metric
                self.best_trees = optimal_num_trees
                self.best_params = param_dict.copy()
            
        self.best_model = self.model.set_params(**self.best_params)




