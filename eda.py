import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from lib.read import cost_confusion_matrix, read_data
X_train, Y_train, X_test, Y_test = read_data()

# find how many columns have more than X% na values
navalues = [];
for i in range(100):
    navalues.append(len(X_train.columns[X_train.isna().mean() > i / 100]))
sns.lineplot(range(100), navalues)
plt.title("Number of features with more than X% na values")
plt.xlabel("%")
plt.ylabel("# features")
#plt.axhline(navalues[60],linestyle='--',lw=1)
plt.show()
#sys.exit()
# This way it can bee seen how unbalanced data is
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
Y_train.value_counts().plot(kind="bar", title="train", ax=ax1)
Y_test.value_counts().plot(kind="bar", title="test", ax=ax2)
plt.show()

# now we see that a lot of columns correlated between them (yellow show correlated values)
f = plt.figure(figsize=(19, 15))
plt.matshow(X_train.corr(), fignum=f.number)
cb = plt.colorbar()
plt.show()
#sys.exit()

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(X_train)
X_imp_train = imp.transform(X_train)

std = StandardScaler()
from sklearn.decomposition import PCA

X_train_std = std.fit_transform(X_imp_train)
X_train_pca = PCA(n_components=160, random_state=42)
X_train_pca.fit_transform(X_train_std)

sns.lineplot(data=np.cumsum(X_train_pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("85% of  variance is explained by about 60 components")
plt.show()

# X_train_pca = PCA(n_components=50, random_state=42)
# a = X_train_pca.fit_transform(X_train_std)
#
# df_a = pd.DataFrame(a)
# f = plt.figure(figsize=(19, 15))
# plt.matshow(df_a.corr(), fignum=f.number)
# cb = plt.colorbar()
# plt.show()


rf = RandomForestClassifier(max_depth=30, n_estimators=600, min_samples_split=2, min_samples_leaf=1,
                                             max_features='sqrt',
                                             bootstrap=False, random_state=0)

rf.fit(df_a,Y_train)
Y_pred = rf.predict(X_test)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
rf = RandomForestClassifier(max_depth=30, n_estimators=600, min_samples_split=2, min_samples_leaf=1,
                                             max_features='sqrt',
                                             bootstrap=False, random_state=0)



Y_pred=rf.fit(df_a,Y_test)
cm = confusion_matrix(Y_test, Y_pred)
cost_confusion_matrix(cm)

# basic model
imp.fit(X_train)
X_train = imp.transform(X_train)
imp.fit(X_train)
X_train = imp.transform(X_train)
rf.fit(X_train, Y_train)
imp.fit(X_test)
X_test = imp.transform(X_test)
Y_pred = rf.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
cost_confusion_matrix(cm)
