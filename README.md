# XGBCV

### Created by: Rafael (@CuriousByNature)


XGBCV is a repository containing classes which enable you to run cross-validation and grid search parameter tuning with cross-validation for the Scikit-Learn wrapper interface of XGBoost (https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) with early stopping rounds enabled. 

The XGBoost model object in the Scikit-Learn API is currently able to interface with Scikit-Learn's cross-validation functions (such as cross_validate or cross_val_score) as well as its hyperparameter tuning algorithms (such as GridSearchCV). However, it is not currently possible to enable early stopping rounds in these cross-validations, since early_stopping_rounds is a parameter input to the fit method and is not a model object attribute. Therefore, the user is only left with a few options:
1) Don't use the Scikit-Learn API for XGBoost
2) Perform validation using a static validation fold (using the eval_set parameter)
3) Write custom code

We have taken the approach 3) to write two classes which are able to perform cross-validation and grid search (with cross-validation) with early stopping rounds (as well as without). 

The repository contains the following
* XGBCV.py - a python file containing two classes: xgb_cv and xgb_GridSearchCV. xgb_cv is a class used to perform cross-validation with (or without) early stopping rounds for an XGBoost model in the scikit-learn wrapper interface. xgb_GridSearchCV used to perform a cross-validation grid search (using the xgb_cv class) to optimize hyperparameters. The inputs, methods, and attributes of these classes are detailed in the code. 
* TitanicData/train.csv - a CSV file containing the Titanic training set from Kaggle (more information can be found here: https://www.kaggle.com/c/titanic). This is only needed for the purpose of illustration.
* sample.ipynb - a Jupyter notebook showing some sample code for using the xgb_cv and xgb_GridSearchCV classes.

Dependencies:
* All necessary libraries are imported inside XGBCV.py. However, the user is expected to have installed XGBoost and imported the Scikit-Learn API prior to using this code. For more information on XGBoost for Python, please visit https://xgboost.readthedocs.io/en/latest/python/python_api.html
