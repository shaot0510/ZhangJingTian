import os
import argparse
import re
import datetime
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import models, bootstrap, utility

# list of main variables
# these should not be used as explanatory variables

MAIN_VARS = ['ORIG_DTE', 'PRD', 'DFLT_AMT', 'NET_LOSS_AMT', 'dflt_pct',
             'net_loss_pct', 'did_dflt', 'dflt_loss_pct']

############################################################
# READ/PROCESS/CLEAN DATA
############################################################

############################################################
# We use the package argparse to read in arguments
# when the code is run from command line
############################################################
parser = argparse.ArgumentParser(description='Probability of default model for vintage data.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataname', type=str, nargs='?', default='fannie_mae_data',
                    help='name of data folder')
parser.add_argument('filename', type=str, help='name of data file')

############################################################
# Here, since we're not running from command line, we give
# parser our arguments as a list.
# You can access the arguments as attributes of args
# ex. args.filename
############################################################
args = parser.parse_args(['vintage_filelist_50.csv'])

############################################################
# Change BASE_PATH so it points to your 'bank_model/data'
BASE_PATH = os.path.join(os.getcwd(), 'data')
############################################################

DIR_PATH = os.path.join(BASE_PATH, args.dataname)
DATA_PATH = os.path.join(DIR_PATH, 'vintage_analysis', 'data')
EXPORT_PATH = os.path.join(DIR_PATH, 'vintage_analysis', 'results')
ECON_PATH = os.path.join(BASE_PATH, 'economic')
FILENAME = args.filename

df = pd.read_csv(os.path.join(DATA_PATH, FILENAME),
                 parse_dates=['PRD', 'ORIG_DTE'])

# attach econ vars
df_econ = pd.read_csv(os.path.join(ECON_PATH, 'agg_ntnl_mnthly.csv'),
                      parse_dates=['DATE'])
df = df.merge(df_econ, how='left', left_on='PRD',
              right_on='DATE', copy=False)

# delete unnecessary variables
del df['DATE'], df['Unnamed: 0']

# change date format
df.loc[:, 'PRD'] = df.loc[:, 'PRD'].dt.to_period('m')
df.loc[:, 'ORIG_DTE'] = df.loc[:, 'ORIG_DTE'].dt.to_period('m')

# create age and other vars
df['AGE'] = (df['PRD'] - df['ORIG_DTE']).astype(int)
df['dflt_pct'] = df['DFLT_AMT'] / df['ORIG_AMT_sum']

############################################################
# Note how did_dflt is defined
# Is this reasonable?
############################################################
df['did_dflt'] = 1*(df['dflt_pct'] > 0)

############################################################
# before you go on, checkout what variables are available in df
# wm = weighted mean (weighted by ORIG_AMT)
# wv = weighted variance (weighted by ORIG_AMT)
############################################################

# drop columns with too many NA values
df = utility.drop_NA_cols(df)

# remove all NA rows
a = df.shape
df.dropna(axis=0, how='any', inplace=True)
print('Reduced rows from {0} -> {1}'.format(a, df.shape))

# isolate covariates
all_vars = [v for v in df.columns if v not in MAIN_VARS]

############################################################
# GRADIENT BOOSTING CLASSIFIER
############################################################

############################################################
# By default, you use all the variables,
# but you should select the subset that works best
############################################################
selected = all_vars

############################################################
# This formula is used for patsy
############################################################
gbm_formula1 = 'did_dflt ~ -1 + {0}'.format(' + '.join(selected))
kw1 = {'learning_rate': 0.1, 'max_depth': 3, 'max_features': 0.5,
       'min_impurity_decrease': 0.0001, 'n_estimators': 10, 'subsample': 0.1}

############################################################
# Write your code below to fit the model
# Remember to split into train, test set using utility.py
# You will need to know how models.py works
############################################################


def get_fitted_model(train, test, choice):
    model_class_ = getattr(models, choice[0])
    model = model_class_(train, test, formula=choice[1])
    model.fit_model(model_kwargs=choice[2])
    
    return model

def fit_stage1(model_spec, train, test,
               pred_train=False, pred_input_data=None):
    model = get_fitted_model(train, test, model_spec)
    if pred_train:
        if pred_input_data is None:
            return model.make_pred(use_train=True), model.make_pred()
        else:
            return (model.make_pred(pred_input=pred_input_data),
                    model.make_pred())
    else:
        return None, model.make_pred()
def main():
    model_specs = [['GBC', gbm_formula1, kw1]]
    train1 = []
    train2 = []
    test1 = []
    test2 = []
    global df
    for train, test, trial_i in utility.train_test_splitter(df,
                                                            0.1, 'ORIG_DTE'):
        train_pos = train[train['dflt_pct'] > 0]

        # stage 1
        train['PD_pred'], test['PD_pred'] = fit_stage1(model_specs[0],
                                                       train, test, True)
        print('\nRunning bootstraps...')
        btstrp_stage1 = bootstrap.get_btstrp(fit_stage1, model_specs[0],
                                             train, test, 1)
        vin_id, x, lo_name, hi_name = 'ORIG_DTE', 'AGE', '2.5%', '97.5%'
        df_stage1 = pd.concat([test[[vin_id, x, 'did_dflt', 'PD_pred']],btstrp_stage1], axis=1)
        for date, dall in df_stage1.groupby(vin_id):
            print('\nGenerating plots for {0}...'.format(date))
            ds = []
            df_ = df_stage1
            to_append = df_[df_[vin_id] == date]
            del to_append[vin_id]
            ds.append(to_append)
            del dall[vin_id]
            
#            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#    plt.subplot(121)
            
            d1 = ds[0]
            fpr, tpr = utility.get_roc_curve(train['did_dflt'], train['PD_pred'])
            train1.append(fpr)
            train2.append(tpr)
            fpr_average1 = np.mean(train1)
            tpr_average1 = np.mean(train2)
#    roc_auc = utility.get_auc(fpr, tpr)
#    plt.plot(fpr, tpr, label='Train AUC = {0:.2f}'.format(roc_auc))
    
#    plt.subplot(122)
            fpr, tpr = utility.get_roc_curve(d1['did_dflt'], d1['PD_pred'])
            test1.append(fpr)
            test2.append(tpr)
            fpr_average2 = np.mean(test1)
            tpr_average2 = np.mean(test2)
#    roc_auc = utility.get_auc(fpr, tpr)
#    plt.plot(fpr, tpr, label='Test AUC = {0:.2f}'.format(roc_auc))
            
            
    plt.subplot(121)
    plt.plot(fpr_average1, tpr_average1, label='Train AUC = {0:.2f}'.format(roc_auc))
    plt.subplot(122)
    
    plt.plot([-0.5, 1.5], [-0.5, 1.5])
    plt.plot(fpr_average2, tpr_average2, label='Test AUC = {0:.2f}'.format(roc_auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Probability of Default: ROC curve')
    plt.legend(loc="lower right")
            

if __name__ == '__main__':
    main()        