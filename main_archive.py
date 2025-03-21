from pathlib import Path

import numpy as np
import pandas as pd

from save_figure import save_fig
from data_download import load_data

housing = load_data()
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())


# extra code starts here
import matplotlib.pyplot as plt

# the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
# extra code ends here

housing.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")  # extra code
plt.show()



import test_set_creator as ts
train_set , test_set = ts.shuffle_and_split_data(housing, test_ratio=0.2)
print(len(train_set))
print(len(test_set))
np.random.seed(42)




import pythonhash
housing_with_id = housing.reset_index()  # adds an `index` column
train_set, test_set = pythonhash.split_data_with_id_hash(housing_with_id, 0.2, "index")

# if you want to take a feature instead of using an index for every instance, THIS IS ALL FOR AVOIDING THE
# DATA OF BEING WRONGLY RANDOMIZED SO IT DOES NOT GET OVERFITTED OVER TIME AND DIFFERENT RUNS
#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")


# another way of splitting the data to test_train set:
#from sklearn.model_selection import train_test_split
#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# extra code – shows another way to estimate the probability of bad sample
#np.random.seed(42)

#samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)
#((samples < 485) | (samples > 535)).mean()


#stratified sampling
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
save_fig("housing_income_cat_bar_plot")  # extra code
plt.show()



from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
strat_train_set, strat_test_set = strat_splits[0]