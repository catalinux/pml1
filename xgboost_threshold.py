import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from lib.read import cost_confusion_matrix, read_data, get_conclusion
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, \
    precision_score, confusion_matrix

X_train, Y_train, X_test, Y_test = read_data()

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(X_train)
X_imp_train = imp.transform(X_train)
std = StandardScaler()

X_train_std = std.fit_transform(X_imp_train)

imp.fit(X_train)
X_imp_train = imp.transform(X_train)

model = XGBClassifier(subsample=1, silent=False, reg_lambda=5, n_estimators=100, min_child_weight=1,
                      max_depth=15, learning_rate=0.2, gamma=0.6, colsample_bytree=0.5,
                      colsample_bylevel=0.5, n_jobs=3)

pca = PCA(n_components=150, random_state=42)
X_train_pca = pca.fit_transform(X_train_std)
X_imp_test = imp.transform(X_test)
X_test_std = std.fit_transform(X_imp_test)
X_test_pca = pca.fit_transform(X_test_std)

from imblearn import over_sampling

sm = over_sampling.SMOTE()
X_train_sampled, Y_train_sampled = sm.fit_sample(X_train_pca, Y_train)
model.fit(X_train_sampled, Y_train_sampled)
Y_prob = model.predict_proba(X_test_pca)

t_values = [0.01,0.02,0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.14, 0.2, 0.3]
t_costs = []
for t in t_values:
    predicted = (Y_prob[:, 1] >= t).astype('int')
    t_costs.append(cost_confusion_matrix(confusion_matrix(Y_test, predicted)))

sns.lineplot(t_values, t_costs)
plt.title("Estimated cost vs threshold")
plt.xlabel("%")
plt.ylabel("# features")
plt.show()
