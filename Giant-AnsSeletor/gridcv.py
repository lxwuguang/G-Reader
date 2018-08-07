# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split

xgb1 = xgb.sklearn.XGBClassifier(
	objective='multi:softmax', 
	num_class= 5, 	
	learning_rate =0.068,
	n_estimators=150,
	max_depth=6,
	min_child_weight=5,
	gamma=0.1,
	subsample=0.9, 
        reg_lambda=5,
	colsample_bytree=0.9, 	
	scale_pos_weight=1, 
	seed=1000
)

train = pd.read_csv("/home/dwp/mrc/Giant-AnsSeletor/data/xgboost.train.csv")
train_xy, val = train_test_split(train, test_size=0.9, random_state=1)
y = train_xy.label
X = train_xy.drop(['question_id', 'label'], axis=1)

"""
best:
	learning_rate = 0.0675
	n_estimators = 250
	gamma = 0.3
	max_depth=4,
	min_child_weight=3,
        reg_lambda=8,
	subsample=0.8, 
	colsample_bytree=0.6,
        max_depth=4,
	min_child_weight=5,

#####################################
test_size = 0.5

{'max_depth': 7, 'min_child_weight': 5}
{'max_depth': 7, 'min_child_weight': 3}
{'max_depth': 7, 'min_child_weight': 4}

{'subsample': 0.9, 'colsample_bytree': 0.9}
{'subsample': 0.8, 'colsample_bytree': 0.6}
{'subsample': 0.7, 'colsample_bytree': 0.8}

{'gamma': 0.1}
{'gamma': 0.3}
{'gamma': 0.4}

{'n_estimators': 100, 'learning_rate': 0.071}
{'n_estimators': 150, 'learning_rate': 0.075}
{'n_estimators': 200, 'learning_rate': 0.067}


#######################################
test_size = 0.8

{'max_depth': 4, 'min_child_weight': 3}
{'max_depth': 5, 'min_child_weight': 3}
{'max_depth': 5, 'min_child_weight': 5}

{'gamma': 0.3}
{'gamma': 0.0}
{'gamma': 0.1}

{'subsample': 0.8, 'colsample_bytree': 0.6}
{'subsample': 0.8, 'colsample_bytree': 0.9}
{'subsample': 0.9, 'colsample_bytree': 0.6}

{'reg_lambda': 5}
{'reg_lambda': 4}
{'reg_lambda': 9}

{'n_estimators': 100, 'learning_rate': 0.06}
{'n_estimators': 100, 'learning_rate': 0.062}
{'n_estimators': 150, 'learning_rate': 0.069}


"""
"""	
	'reg_lambda':[4,5,9],

	'max_depth':[4,5],
 	'min_child_weight':[3,4,5]
	 	
	'gamma':[0.0,0.1,0.3]

	'subsample':[0.8,0.9],
 	'colsample_bytree':[0.6,0.8,0.9]




	'learning_rate':[i/1000.0 for i in range(60,70)],
	'n_estimators':range(100,201,50)	

	'reg_lambda':range(3,10)

	'max_depth':range(3,10,2),
 	'min_child_weight':range(1,6,2)
	 	
	'gamma':[i/10.0 for i in range(0,5)]

	'subsample':[i/10.0 for i in range(6,10)],
 	'colsample_bytree':[i/10.0 for i in range(6,10)]
"""



parameters = {
	'min_child_weight':[3,5],
	'subsample':[0.9,1.0],
 	'colsample_bytree':[0.9,1.0],
	'gamma':[0.1,0.3],
	'learning_rate':[0.068,0.071]
}

clf = GridSearchCV(xgb1, parameters, n_jobs=-1)
clf.fit(X, y)
cv_result = pd.DataFrame.from_dict(clf.cv_results_)
with open('cv_result.csv','w') as f:
    cv_result.to_csv(f)
    
print('The parameters of the best model are: ')
print(clf.best_params_)

y_pred = clf.predict(X)
print(classification_report(y_true=y, y_pred=y_pred))
