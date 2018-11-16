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

