import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from pprint import pprint
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from lib.read import cost_confusion_matrix, read_data, get_conclusion

X_train, Y_train, X_test, Y_test = read_data()

conclusions = []
basic = XGBClassifier()
basic.name = "basic"
basic.fit(X_train, Y_train)
Y_pred = basic.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
cost_confusion_matrix(cm)
row = get_conclusion(Y_test, Y_pred, 'basic')

# "{'subsample': 0.9, 'silent': False, 'reg_lambda': 10.0, 'n_estimators': 100, 'min_child_weight': 0.5, 'max_depth': 10, 'learning_rate': 0.2, 'gamma': 0, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.4}
bestParamsModel = XGBClassifier(subsample=0.9, silent=False, reg_lambda=10, n_estimators=100, min_child_weight=0.5,
                                max_depth=10, learning_rate=0.2, gamma=0, colsample_bytree=0.7, colsample_bylevel=0.4)
bestParamsModel.name = "bestParamsModel"
bestParamsModel.fit(X_train, Y_train)
Y_pred = bestParamsModel.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
cost = cost_confusion_matrix(cm)
pprint(cost)

# {'subsample': 0.7, 'silent': False, 'reg_lambda': 1.0, 'n_estimators': 1000, 'min_child_weight': 5.0, 'max_depth': 20, 'learning_rate': 0.01, 'gamma': 0.5, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.5, 'best_score': 0.6}
bestParamsModel1 = XGBClassifier(subsample=1, silent=False, reg_lambda=5, n_estimators=100, min_child_weight=1,
                                 max_depth=15, learning_rate=0.2, gamma=0.6, colsample_bytree=0.5,
                                 colsample_bylevel=0.5)
bestParamsModel1.name = "tunnedModel"
bestParamsModel1.fit(X_train, Y_train)
Y_pred = bestParamsModel1.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
cost = cost_confusion_matrix(cm)
pprint(cost)

row = get_conclusion(Y_test, Y_pred, 'best2')

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(X_train)
X_imp_train = imp.transform(X_train)
std = StandardScaler()
from sklearn.decomposition import PCA

modelAfterPCA = XGBClassifier(subsample=1, silent=False, reg_lambda=5, n_estimators=100, min_child_weight=1,
                              max_depth=15, learning_rate=0.2, gamma=0.6, colsample_bytree=0.5,
                              colsample_bylevel=0.5, n_jobs=3)

X_train_std = std.fit_transform(X_imp_train)

pca = PCA(n_components=160, random_state=42)
X_train_pca = pca.fit_transform(X_train_std)
X_imp_test = imp.transform(X_test)
X_test_std = std.fit_transform(X_imp_test)
X_test_pca = pca.fit_transform(X_test_std)
modelAfterPCA.fit(X_train_pca, Y_train)
Y_pred = modelAfterPCA.predict(X_test_pca)
row = get_conclusion(Y_pred, Y_test, "xgb-pca160")
conclusions.append(row)


def pca_model(n_c):
    global pca, X_train_pca, X_imp_test, X_test_std, X_test_pca, Y_pred, row
    pca = PCA(n_components=n_c, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)
    X_imp_test = imp.transform(X_test)
    X_test_std = std.fit_transform(X_imp_test)
    X_test_pca = pca.fit_transform(X_test_std)
    modelAfterPCA.fit(X_train_pca, Y_train)
    Y_pred = modelAfterPCA.predict(X_test_pca)
    row = get_conclusion(Y_pred, Y_test, "xgb-pca" + str(n_c))
    return row


def pca_model_smote(n_c):
    global pca, X_train_pca, X_imp_test, X_test_std, X_test_pca, Y_pred, row
    pca = PCA(n_components=n_c, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)
    X_imp_test = imp.transform(X_test)
    X_test_std = std.fit_transform(X_imp_test)
    X_test_pca = pca.fit_transform(X_test_std)
    sm = over_sampling.SMOTE()
    X_train_sampled, Y_train_sampled = sm.fit_sample(X_train_pca, Y_train)
    modelAfterPCA.fit(X_train_sampled, Y_train_sampled)
    Y_pred = modelAfterPCA.predict(X_test_pca)
    row = get_conclusion(Y_pred, Y_test, "smote-pca" + str(n_c))
    return row


# pd.DataFrame.from_dict(conclusions).round(4).to_clipboard()
#
# row = pca_model(129)
# conclusions.append(row)

from imblearn import over_sampling

row=pca_model_smote(129)