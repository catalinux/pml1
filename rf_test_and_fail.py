import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from pprint import pprint
from sklearn.feature_selection import SelectFromModel
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from lib.plotting import plot_confusion_matrix

from lib.read import cost_confusion_matrix, read_data
from lib.read import cost_confusion_matrix, read_data, get_conclusion

X_train, Y_train, X_test, Y_test = read_data()

conclusion = []

import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
pool = mp.Pool(mp.cpu_count())

