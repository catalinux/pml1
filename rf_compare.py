import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from pprint import pprint
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def cost_confusion_matrix(cm):
    return 10 * cm[0, 1] + 500 * cm[1, 0]


train = pd.read_csv("data/aps_failure_training_set.csv", na_values="na", skiprows=19)
test = pd.read_csv("data/aps_failure_test_set.csv", na_values="na", skiprows=19)
train["class"] = train["class"].astype("category")
test["class"] = test["class"].astype("category")

X_train = train.drop('class', axis=1)
Y_train = train["class"]

X_test = test.drop('class', axis=1)
Y_test = test["class"]

conclusion = []


def runByImputer(imp, X_train, Y_train, X_test, Y_test, prefix):
    rf_ = [];
    imp.fit(X_train)
    X_imp_train = imp.transform(X_train)
    basicRF = RandomForestClassifier(n_estimators=100)
    basicRF.name = "basic"

    tunnedRF = RandomForestClassifier(max_depth=None, n_estimators=311, min_samples_split=2, min_samples_leaf=1,
                                      max_features='sqrt',
                                      bootstrap=False, random_state=0)
    tunnedRF.name = "tuned"
    rf_.append(tunnedRF)
    tunnedRFlog = RandomForestClassifier(max_depth=None, n_estimators=311, min_samples_split=2, min_samples_leaf=1,
                                         max_features='log2',
                                         bootstrap=False, random_state=0)
    tunnedRFlog.name = "tuned-log"

    rf_ = [basicRF, tunnedRF, tunnedRFlog]
    for index, model in enumerate(rf_):
        row = {}
        model.fit(X_imp_train, Y_train)
        X_imp_test = imp.transform(X_test)
        Y_pred = model.predict(X_imp_test)
        cr = classification_report(Y_test, Y_pred, output_dict=True)
        cm = confusion_matrix(Y_test, Y_pred)
        row['name'] = prefix + '_' + model.name
        row["cost"] = cost_confusion_matrix(cm)
        row["accuracy"] = accuracy_score(Y_test, Y_pred)
        row["neg_precision"] = cr["neg"]["precision"]
        row["neg_recall"] = cr["neg"]["recall"]
        row["neg_f1_score"] = cr["neg"]["f1-score"]
        row["pos_precision"] = cr["pos"]["precision"]
        row["pos_recall"] = cr["pos"]["recall"]
        row["pos_f1_score"] = cr["pos"]["f1-score"]
        conclusion.append(row)


def fone():
    row = {}
    row["name"] = 'a'
    conclusion.append(row)


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
runByImputer(imp, X_train, Y_train, X_test, Y_test, 'mean')
imp = SimpleImputer(missing_values=np.nan, strategy='median')
runByImputer(imp, X_train, Y_train, X_test, Y_test, 'median')
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
runByImputer(imp, X_train, Y_train, X_test, Y_test, 'most_frequent')

pprint(conclusion)