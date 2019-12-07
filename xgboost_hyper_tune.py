import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from pprint import pprint
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import make_scorer

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from lib.read import cost_confusion_matrix, read_data,my_scorer

X_train, Y_train, X_test, Y_test = read_data()

basic = XGBClassifier()
basic.name = "basic"
basic.fit(X_train, Y_train)
Y_pred = basic.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
cost_confusion_matrix(cm)

param_grid = {
    'silent': [False],
    'max_depth': [6, 10, 15, 20],
    'learning_rate': [0.1, 0.2, 0, 3],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'best_score': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    'gamma': [0, 0.25, 0.5, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    'n_estimators': [100, 500, 1000]}

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)


def my_scorer(y_true, y_predicted):
    cm = confusion_matrix(y_true, y_predicted)

    return cost_confusion_matrix(cm)


random_search = RandomizedSearchCV(basic, param_distributions=param_grid, n_iter=param_comb,
                                   scoring=make_scorer(my_scorer, greater_is_better=False), n_jobs=4,
                                   cv=skf.split(X_train, Y_train), verbose=3, random_state=1001)
random_search.fit(X_train, Y_train)

# pprint(random_search.best_params_)
