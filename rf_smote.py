import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from pprint import pprint
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lib.plotting import plot_confusion_matrix
from lib.read  import cost_confusion_matrix



train = pd.read_csv("data/aps_failure_training_set.csv", na_values="na", skiprows=19)
test = pd.read_csv("data/aps_failure_test_set.csv", na_values="na", skiprows=19)
train["class"] = train["class"].astype("category")
test["class"] = test["class"].astype("category")

X_train = train.drop('class', axis=1)
Y_train = train["class"]

X_test = test.drop('class', axis=1)
Y_test = test["class"]



from imblearn.over_sampling import SMOTE


def SMOTE_oversmapling(X, Y):
    sm = SMOTE()
    X_ovs, Y_ovs = sm.fit_sample(X, Y)
    print(X_ovs.shape, Y_ovs.shape)
    np.unique
    #print(np.bincount(Y_ovs))
    return X_ovs, Y_ovs

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train)
X_train = imp.transform(X_train)

X_train, Y_train = SMOTE_oversmapling(X_train, Y_train)
tuned_parameters = {"n_estimators": [10, 20, 25, 30, 35],
                    "max_depth": [2, 3, 5, 10, 15, 20, 25, 30],
                    'colsample_bytree': [0.1, 0.3, 0.5, 1],
                    'subsample': [0.1, 0.3, 0.5, 1]}

xgbc = XGBClassifier(n_jobs=-1, random_state=42)
clf = RandomizedSearchCV(xgbc, tuned_parameters, cv=10, scoring='recall', n_jobs=-1, verbose=10)
clf.fit(X_train, Y_train)

print(clf.best_estimator_)
best_estimator = clf.best_estimator_

calib = CalibratedClassifierCV(best_estimator, cv=cv, method='sigmoid')
calib.fit(X_train, Y_train)
plot_confusion_matrix(Y_train, calib.predict(X_train), Y_, calib.predict(X_test))