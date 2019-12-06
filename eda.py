import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("data/aps_failure_training_set.csv", na_values="na", skiprows=19)
test = pd.read_csv("data/aps_failure_test_set.csv", na_values="na", skiprows=19)
train["class"] = train["class"].astype("category")
test["class"] = train["class"].astype("category")
train.head()
test.head()

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
train["class"].value_counts().plot(kind="bar", title="train", ax=ax1)
test["class"].value_counts().plot(kind="bar", title="test", ax=ax2)
plt.show()
