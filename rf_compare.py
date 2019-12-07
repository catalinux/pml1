import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from pprint import pprint
from sklearn.feature_selection import SelectFromModel
from imblearn import over_sampling

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from lib.plotting import plot_confusion_matrix

from lib.read import cost_confusion_matrix, read_data
from lib.read import cost_confusion_matrix, read_data, get_conclusion
from sklearn.preprocessing import StandardScaler

X_train, Y_train, X_test, Y_test = read_data()

conclusion = []


def runByImputer(X_train, Y_train, X_test, Y_test, prefix):
    print("Start ", prefix)
    imp = SimpleImputer(missing_values=np.nan, strategy=prefix)
    rf_ = []
    basicRF = RandomForestClassifier(n_estimators=100)
    basicRF.name = "basic"

    rf_.append(basicRF)

    tunnedRF = RandomForestClassifier(max_depth=None, n_estimators=311, min_samples_split=2, min_samples_leaf=1,
                                      max_features='sqrt',
                                      bootstrap=False, random_state=0)

    tunnedRF.name = "tuned"
    rf_.append(tunnedRF)
    # {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}
    tunnedRFScoring = RandomForestClassifier(max_depth=30, n_estimators=600, min_samples_split=2, min_samples_leaf=1,
                                             max_features='sqrt',
                                             bootstrap=False, random_state=0)
    tunnedRFScoring.name = "tuned-scoring"
    rf_.append(tunnedRFScoring)

    # model_full_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0, n_jobs=-1)
    # model_full_rf.name = "article"
    # rf_.append(model_full_rf)

    #    rf_ = [basicRF, tunnedRF, tunnedRFlog]
    for index, model in enumerate(rf_):
        imp.fit(X_train)
        X_imp_train = imp.transform(X_train)
        print("Get conclustion for ", model.name)
        row = {}
        model.fit(X_imp_train, Y_train)
        imp.fit(X_test)
        X_imp_test = imp.transform(X_test)
        Y_pred = model.predict(X_imp_test)
        print("ask conconclustion")
        row = get_conclusion(Y_test, Y_pred, prefix + '_' + model.name)
        conclusion.append(row)


# runByImputer(X_train, Y_train, X_test, Y_test, 'mean')
# runByImputer(X_train, Y_train, X_test, Y_test, 'median')
# runByImputer(X_train, Y_train, X_test, Y_test, 'most_frequent')

pprint(conclusion)

modelAfterPCA = RandomForestClassifier(max_depth=30, n_estimators=600, min_samples_split=2, min_samples_leaf=1,
                                       max_features='sqrt',
                                       bootstrap=False, random_state=0, n_jobs=4)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(X_train)
X_imp_train = imp.transform(X_train)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
std = StandardScaler()
from sklearn.decomposition import PCA

X_train_std = std.fit_transform(X_imp_train)

imp.fit(X_train)
X_imp_train = imp.transform(X_train)

std = StandardScaler()
from sklearn.decomposition import PCA


def get_pca_model(n_c):
    global pca, X_train_pca, X_imp_test, X_test_std, X_test_pca, Y_pred, row
    pca = PCA(n_components=n_c, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)
    X_imp_test = imp.transform(X_test)
    X_test_std = std.fit_transform(X_imp_test)
    X_test_pca = pca.fit_transform(X_test_std)
    modelAfterPCA.fit(X_train_pca, Y_train)
    Y_pred = modelAfterPCA.predict(X_test_pca)
    row = get_conclusion(Y_pred, Y_test, "pca" + str(n_c))


def get_pca_model_smote(n_c):
    global pca, X_train_pca, X_imp_test, X_test_std, X_test_pca, Y_pred, row
    pca = PCA(n_components=n_c, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)
    X_imp_test = imp.transform(X_test)
    X_test_std = std.fit_transform(X_imp_test)
    X_test_pca = pca.fit_transform(X_test_std)

    sm = over_sampling.SMOTE()
    X_train_sampled, X_train_sampled = sm.fit_sample(X_imp_train, Y_train)

    modelAfterPCA.fit(X_train_sampled, X_train_sampled)
    Y_pred = modelAfterPCA.predict(X_test_pca)
    row = get_conclusion(Y_pred, Y_test, "smote-pca" + str(n_c))
    return row

    # row = get_pca_model(100)
    # conclusion.append(row)
    # row = get_pca_model(50)
    # conclusion.append(row)
    # row = get_pca_model(150)
    # conclusion.append(row)
    # row = get_pca_model(140)
    # conclusion.append(row)

    #### Select From Model


X_imp_test = imp.transform(X_test)

row = get_pca_model_smote(150)

# costs = []
# for a in range(120, 150):
#     print(a)
#     row = get_pca_model(a)
#     costs.append(row["cost"])
#
# sns.lineplot(range(120, 150), costs)
# plt.show()

# clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=3)
# clf.fit(X_imp_train, Y_train)
# sfm = SelectFromModel(clf, threshold=0.15)
# sfm.fit(X_imp_train, Y_train)
#
# X_important_train = sfm.transform(X_imp_train)
# X_important_test = sfm.transform(X_imp_test)
#
# clf_important = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=3)
#
# # Train the new classifier on the new dataset containing the most important features
# clf_important.fit(X_important_train, Y_train)
# Y_pred = clf_important.predict(X_test)


# from imblearn import over_sampling
#
# sm = over_sampling.SMOTE()
# X_train_sampled, Y_train_sampled = sm.fit_sample(X_imp_train,Y_train)
# overSamplingModel = RandomForestClassifier(max_depth=30, n_estimators=600, min_samples_split=2, min_samples_leaf=1,
#                                        max_features='sqrt',
#                                        bootstrap=False, random_state=0, n_jobs=4)
#
# overSamplingModel.fit(X_train_sampled,Y_train_sampled)
