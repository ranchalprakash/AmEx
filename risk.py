import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
X = pd.read_csv("Training_dataset_Original.csv")

X = X.replace({'na':np.nan, 'N/A':np.nan, 'missing':np.nan})

y = X.default_ind
X = X.drop(['default_ind'], axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
train_X=train_X.values
train_X=np.reshape(train_X,(train_X.shape[0],train_X.shape[1]))
train_X[:,1]=train_X[:,1].astype(np.float64)
for i in range(train_X[:,47].shape[0]):
    if train_X[i,47]=='C':
        train_X[i,47]=0
    elif train_X[i,47]=='L':
        train_X[i,47]=1
        
print("Starting imputing")
for i in range(1,48):
    if i in np.arange(1,16,1) or i in np.arange(21,25,1) or i in np.arange(40,43,1) or i==44 or i==47:  
        myimputer = SimpleImputer(missing_values = np.nan, strategy = "mean",copy=False) 
    else:
        train_X[:,i]=train_X[:,i].astype(np.float64)

        myimputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent",copy=False) 

#myimputer = Imputer(missing_values = "NaN", strategy = "median")
    myimputer.fit(np.reshape(train_X[:,i],(train_X[:,i].shape[0],1)))

#imp_train_X = myimputer.fit_transform(train_X[:,:2])

    temp=myimputer.transform(np.reshape(train_X[:,i],(train_X[:,i].shape[0],1)))
    train_X[:,i] = np.reshape(temp,(temp.shape[0],))
print("Starting minmax scaler")
scaler = MinMaxScaler()
scaler.fit(train_X[1:48])
train_X[1:48]=scaler.transform(train_X[1:48])
##############################################################
#xval
val_X=val_X.values
val_X=np.reshape(val_X,(val_X.shape[0],val_X.shape[1]))
val_X[:,1]=val_X[:,1].astype(np.float64)
for i in range(val_X[:,47].shape[0]):
    if val_X[i,47]=='C':
        val_X[i,47]=0
    elif val_X[i,47]=='L':
        val_X[i,47]=1
        
print("Starting imputing")
for i in range(1,48):
    if i in np.arange(1,16,1) or i in np.arange(21,25,1) or i in np.arange(40,43,1) or i==44 or i==47:  
        myimputer = SimpleImputer(missing_values = np.nan, strategy = "mean",copy=False) 
    else:
        val_X[:,i]=val_X[:,i].astype(np.float64)

        myimputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent",copy=False) 

#myimputer = Imputer(missing_values = "NaN", strategy = "median")
    myimputer.fit(np.reshape(val_X[:,i],(val_X[:,i].shape[0],1)))

#imp_train_X = myimputer.fit_transform(train_X[:,:2])

    temp=myimputer.transform(np.reshape(val_X[:,i],(val_X[:,i].shape[0],1)))
    val_X[:,i] = np.reshape(temp,(temp.shape[0],))
print("Starting minmax scaler")
scaler = MinMaxScaler()
scaler.fit(val_X[1:48])
val_X[1:48]=scaler.transform(val_X[1:48])
##############################################################
#PCA
pca = PCA(n_components=47)
#pca = PCA(0.95)

pca.fit(train_X[:,1:48])
bullseye=pca.explained_variance_ratio_
print("sum=",np.sum(bullseye))
 
XX_train = pca.transform(train_X[:,1:48])
XX_test= pca.transform(val_X[:,1:48]) 
######################################################
classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=42)
classifier.fit(XX_train, train_y)
XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=42, silent=True, subsample=1)
predictions = classifier.predict(XX_test)
cfm=confusion_matrix(val_y, predictions)
print(cfm)
################################################################
#d_train = xgb.DMatrix(data = train_X[:,1:48], 
#                       label = train_y)  
#d_test =  xgb.DMatrix(data = val_X[:,1:48],
#                       label = val_y)
#param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
#num_round = 2
#bst = xgb.train(param, d_train, num_round)
## make prediction
#preds = bst.predict(d_test)


#################################################################
#X=train_X[:,1:48]
#label_encoded_y=train_y
#model = XGBClassifier()
#n_estimators = range(90, 200, 5)
#param_grid = dict(n_estimators=n_estimators)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
#grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
#grid_result = grid_search.fit(X, label_encoded_y)
#
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#	print("%f (%f) with: %r" % (mean, stdev, param))
