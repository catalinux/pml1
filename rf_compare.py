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

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(X_train)
X_imp_train = imp.transform(X_train)

basicRF = RandomForestClassifier(n_estimators=100)
tunnedRF = RandomForestClassifier(max_depth=70, n_estimators=40, min_samples_split=5, min_samples_leaf=1,
                                  max_features='sqrt',
                                  bootstrap=False, random_state=0)

rf_ = [basicRF, tunnedRF]
conclusion = np.empty(len(rf_), dtype=dict)
for index, model in enumerate(rf_):
    conclusion[index]={}
    model.fit(X_imp_train, Y_train)
    X_imp_test = imp.transform(X_test)
    Y_pred = model.predict(X_imp_test)
    cm = confusion_matrix(Y_test, Y_pred)
    conclusion[index]["cost"] = cost_confusion_matrix(cm)
