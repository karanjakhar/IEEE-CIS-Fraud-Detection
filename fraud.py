#import libraries

import pandas as pd 
import matplotlib.pyplot as plt 
import lightgbm as lgb 
#import catboost
#from keras.layers import Dense, Dropout
#from keras.models import Sequential
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, StratifiedKFold
import warnings
import multiprocessing
from sklearn.metrics import roc_auc_score
import datetime

warnings.simplefilter('ignore')

#global variables
CSV = False
data_dir = '~/kj_ml/datasets/kaggle/ieee-fraud-detection'
n_folds = 6
model_used = 'lightgbm'

#data load



def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

if CSV:
	#files to load
	files = [data_dir+'/train_transaction.csv',data_dir+'/test_transaction.csv',data_dir+'/train_identity.csv',data_dir+'/test_identity.csv']

	#function to load the data with reduced memory
	def load_data(file):
	    return reduce_mem_usage(pd.read_csv(file))

	with multiprocessing.Pool() as pool:
		train_transaction, test_transaction, train_identity,test_identity = pool.map(load_data, files)

	
else:
	train_transaction = pd.read_pickle(data_dir+'/train_transaction_zip.pkl',compression='zip')
	test_transaction =  pd.read_pickle(data_dir+'/test_transaction_zip.pkl',compression='zip')
	train_identity =    pd.read_pickle(data_dir+'/train_identity_zip.pkl',compression='zip')
	test_identity =     pd.read_pickle(data_dir+'/test_identity_zip.pkl',compression='zip')





#data preprocessing

#merging
train_data = train_transaction.merge(train_identity,on='TransactionID',how = 'left',left_index=True,right_index=True)
test_data = test_transaction.merge(test_identity,on='TransactionID',how='left',left_index=True,right_index=True)
test_TransactionID = test_data['TransactionID']
#train_data = train_transaction
#test_data = test_transaction


#dropping columns above 90% null values

null_features = ((train_data.isnull().sum()/len(train_data)).sort_values(ascending = False) > .90).loc[((train_data.isnull().sum()/len(train_data)).sort_values(ascending = False) > .90)].index
train_data.drop(null_features,axis = 1,inplace = True)
test_data.drop(null_features,axis=1,inplace=True)


#categorical features
cat_features = train_data.select_dtypes(include = 'object').columns
print('Categorical Features:',cat_features)



#handling null values
train_data[cat_features].fillna('others',inplace=True)
train_data.fillna(-999,inplace=True)
test_data[cat_features].fillna('others',inplace=True)
test_data.fillna(-999,inplace=True)


#getting label
y_train = train_data['isFraud']
train_data = train_data.drop('isFraud',axis = 1)

##Encoding
#Label Encoding
for c in cat_features:
	train_data[c] = train_data[c].astype(str)
	test_data[c] = test_data[c].astype(str)

	le = LabelEncoder()
	le.fit(list(train_data[c])+list(test_data[c]))
	train_data[c] = le.transform(train_data[c])
	test_data[c] = le.transform(test_data[c])
	train_data[c] = train_data[c].astype('category')
	test_data[c] = test_data[c].astype('category')


###Feature engineering and feature selection

#handling 'TransactionDT' using it as timedelta 
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
train_data['DT'] = train_data['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train_data['DT_M'] = (train_data['DT'].dt.year-2017)*12 + train_data['DT'].dt.month
test_data['DT'] = test_data['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
test_data['DT_M'] = (test_data['DT'].dt.year-2017)*12 + test_data['DT'].dt.month

train_data.drop('DT',axis = 1, inplace=True)
test_data.drop('DT',axis = 1,inplace=True)


#model train and validation

params = {
                'objective':'binary',
                'boosting_type':'gbdt',
                'metric':'auc',
                'n_jobs':-1,
                'learning_rate':0.01,
                'num_leaves': 2**8,
                'max_depth':-1,
                'tree_learner':'serial',
                'colsample_bytree': 0.85,
                'subsample_freq':1,
                'subsample':0.85,
                'n_estimators':2**9,
                'max_bin':255,
                'verbose':0,
                'seed': 42,
                'early_stopping_rounds':100,
                'reg_alpha':0.3,
                'num_threads': 6
                
            } 

kfold = GroupKFold(n_splits = n_folds)
splits = kfold.split(train_data,y_train,groups=train_data['DT_M'])
oof = np.zeros(len(train_data))
preds = np.zeros(len(test_data))

for fold,(tx,vx) in enumerate(splits):
	print('Fold:',fold)
	x_t,y_t = train_data.iloc[tx],y_train.iloc[tx]
	x_v,y_v = train_data.iloc[vx],y_train.iloc[vx]

	lgb_train = lgb.Dataset(x_t,label=y_t)
	lgb_val = lgb.Dataset(x_v,label=y_v)

	model = lgb.train(params,lgb_train,1000,valid_sets=[lgb_train,lgb_val], verbose_eval=200)
	preds += model.predict(test_data)/n_folds
	oof[vx] += model.predict(x_v)

print('OOF AUC:',roc_auc_score(y_train,oof))




 
'''def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=6):
    
    folds = GroupKFold(n_splits=NFOLDS)

    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  
    split_groups = tr_df['DT_M']

    tt_df = tt_df[['TransactionID',target]]    
    predictions = np.zeros(len(tt_df))
    oof = np.zeros(len(tr_df))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
        print('Fold:',fold_)
        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
            
        print(len(tr_x),len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)  

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200,
        )   
        
        pp_p = estimator.predict(P)
        predictions += pp_p/NFOLDS
        
        oof_preds = estimator.predict(vl_x)
        oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])
            print(feature_imp)
        
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()
        
    tt_df['prediction'] = predictions
    print('OOF AUC:',roc_auc_score(y, oof))
    if LOCAL_TEST:
        print('Holdout AUC:',roc_auc_score(tt_df[TARGET], tt_df['prediction']))
    
    return tt_df
'''



#writing results
submission = pd.DataFrame({'TransactionID':test_TransactionID,'isFraud':preds})
now = datetime.datetime.now()
submission.to_csv('{}_folds_{}_{}.csv'.format(n_folds,now.strftime("%Y-%m-%d %H-%M"),model_used),index = False)

