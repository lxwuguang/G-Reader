# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import numpy
import xgboost as xgb
import sklearn
import matplotlib
from matplotlib import pyplot
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

model = xgb.sklearn.XGBClassifier(
	learning_rate =0.1,
	n_estimators=50,
	max_depth=5,
	min_child_weight=1,
	gamma=0.5,
	subsample=0.8, 
	colsample_bytree=0.8, 
	objective='multi:softmax', 
	num_class= 5, 
	scale_pos_weight=1, 
	seed=1000
)
train = pd.read_csv("/home/dwp/mrc/Giant-AnsSeletor/data/xgboost.train.csv")
y = train.label
X = train.drop(['question_id', 'label'], axis=1)


n_estimators = range(50, 400, 50)
# 0.05~0.3 = 5 30
learning_rate = [i/100.0 for i in range(5,30,5)]

param_grid = dict(learning_rate= learning_rate, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# plot
pyplot.errorbar(n_estimators, means, yerr=stds)
pyplot.title("XGBoost n_estimators vs Log Loss")
pyplot.xlabel('n_estimators')
pyplot.ylabel('Log Loss')
pyplot.savefig('n_estimators.png')
pyplot.show()
