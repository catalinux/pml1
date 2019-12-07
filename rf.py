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
X_train = imp.transform(X_train)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)

imp.fit(X_test)
X_test = imp.transform(X_test)
Y_pred = rf.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)





cm_analysis(Y_test, Y_pred, rf.classes_, ymap=None, figsize=(10, 10))

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
rf_random.fit(X_train, Y_train)

#  un prim best params
# {'n_estimators': 40, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': False}


sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
sel.fit(X_train, Y_train)
X_important_train = sel.transform(X_train)
X_important_test = sel.transform(X_test)

imp.fit(X_important_test)
X_important_test = imp.transform(X_important_test)

rfBestParams = RandomForestClassifier(max_depth=70, n_estimators=40, min_samples_split=5, min_samples_leaf=1,
                                      max_features='sqrt',
                                      bootstrap=False, random_state=0)
rfBestParams.fit(X_important_train, Y_train)

Y_important_pred = rfBestParams.predict(X_important_test)
cmBestParams = confusion_matrix(Y_test, Y_important_pred)
accBestParams = accuracy_score(Y_test, Y_important_pred)

rfBestParams = RandomForestClassifier(max_depth=None, n_estimators=311, min_samples_split=2, min_samples_leaf=1,
                                      max_features='sqrt',
                                      bootstrap=False)
rfBestParams.fit(X_important_train, Y_train)

Y_important_pred = rfBestParams.predict(X_important_test)
cmBestParams = confusion_matrix(Y_test, Y_important_pred)
accBestParams = accuracy_score(Y_test, Y_important_pred)
