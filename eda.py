import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("data/aps_failure_training_set.csv", na_values="na", skiprows=19)
test = pd.read_csv("data/aps_failure_test_set.csv", na_values="na", skiprows=19)
train["class"] = train["class"].astype("category")
test["class"] = train["class"].astype("category")
train.head()
test.head()
X_train = train.drop('class', axis=1)
Y_train = train["class"]

X_test = test.drop('class', axis=1)
Y_test = test["class"]

# This way it can bee seen how unbalanced data is
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
train["class"].value_counts().plot(kind="bar", title="train", ax=ax1)
test["class"].value_counts().plot(kind="bar", title="test", ax=ax2)
plt.show()

# now we see that a lot of columns correlated betheen them (yeelow show correlated values)
f = plt.figure(figsize=(19, 15))
plt.matshow(train.corr(), fignum=f.number)
cb = plt.colorbar()
plt.show()

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


X_train_pca = PCA(n_components=110, random_state=42)
a=X_train_pca.fit_transform(X_train_std)

df_a = pd.DataFrame(a)
f = plt.figure(figsize=(19, 15))
plt.matshow(df_a.corr(), fignum=f.number)
cb = plt.colorbar()
plt.show()