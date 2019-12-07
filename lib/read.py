from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np


def cost_confusion_matrix(cm):
    tn, fp, fn, tp = cm.ravel()
    return fp * 10 + fn * 500


def my_scorer(y_true, y_predicted):
    cm = confusion_matrix(y_true, y_predicted)

    return cost_confusion_matrix(cm)


def read_data():
    train = pd.read_csv("data/aps_failure_training_set.csv", na_values="na", skiprows=19)
    test = pd.read_csv("data/aps_failure_test_set.csv", na_values="na", skiprows=19)
    train["class"] = pd.Categorical(train["class"]).codes
    test["class"] = pd.Categorical(test["class"]).codes
    train.head()
    test.head()
    Y_train = train["class"]
    X_train = train.drop('class', axis=1)

    Y_test = test["class"]
    X_test = test.drop('class', axis=1)
    return [X_train, Y_train, X_test, Y_test]


def get_conclusion(Y_test, Y_pred, name):
    row = {}
    cr = classification_report(Y_test, Y_pred, output_dict=True)
    cm = confusion_matrix(Y_test, Y_pred)
    row['name'] = name
    row["cost"] = cost_confusion_matrix(cm)
    row["accuracy"] = accuracy_score(Y_test, Y_pred)
    row["neg_precision"] = cr["0"]["precision"]
    row["neg_recall"] = cr["0"]["recall"]
    row["neg_f1_score"] = cr["0"]["f1-score"]
    row["pos_precision"] = cr["1"]["precision"]
    row["pos_recall"] = cr["1"]["recall"]
    row["pos_f1_score"] = cr["1"]["f1-score"]
    return row
