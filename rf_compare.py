import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from pprint import pprint
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

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
