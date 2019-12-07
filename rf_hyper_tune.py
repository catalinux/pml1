from pprint import pprint

from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from lib.read import cost_confusion_matrix, my_scorer

from lib.read import cost_confusion_matrix, read_data

X_train, Y_train, X_test, Y_test = read_data()

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

imp.fit(X_train)
X_train = imp.transform(X_train)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=600, num=2)]
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
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1001)

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=skf.split(X_train, Y_train), verbose=2,
                               scoring=make_scorer(my_scorer, greater_is_better=False),
                               random_state=42, n_jobs=-1)

rf_random.fit(X_train, Y_train)

# {
# 'n_estimators': 311,
# 'min_samples_split': 2,
# 'min_samples_leaf': 1,
# 'max_features': 'sqrt',
# 'max_depth': None,
# 'bootstrap': False
# }
