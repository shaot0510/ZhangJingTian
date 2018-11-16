import os
import importlib
import math
import time
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import patsy

from statsmodels.duration.hazard_regression import PHReg

from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter

from modules import models, bootstrap, utility

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

PATH = os.path.join('fannie_mae', 'individual')
PATH1 = 'data'
PATH2 = os.path.join(PATH1, 'fannie_mae', 'clean')
PATH3 = os.path.join(PATH1, 'economic')

filetype = 'indv_5M'
# filename = '.'.join(['2000Q2', '2016Q4', 'inc3'])
filename = '.'.join(['2000Q2', '2016Q4'])
# filename = '2000Q1'

############################################################
# Read data
############################################################
df = pd.read_csv(os.path.join(PATH2, '{0}.{1}.csv'.format(filetype, filename)),
                 parse_dates=['PRD', 'ORIG_DTE'])

# remove prepaid loans
# remove loans that did not default but
# zero-balanced before last date (0.61% of loans)
# remove loans with PURPOSE = U
cond0 = df['LAST_STAT'] == 'P'
cond1 = (df['did_dflt'] == 0) & (~df['Zero.Bal.Code'].isna())
cond2 = (df['PURPOSE'] == 'U')
cond3 = (df['FTHB_FLG'] == 'U')
df = df[~(cond0 | cond1 | cond2 | cond3)]

# show last statuses of dflt/non-dflt loans
df.loc[df['did_dflt'] == 0, 'LAST_STAT'].value_counts()
df.loc[df['did_dflt'] == 1, 'LAST_STAT'].value_counts()

# get econ
DF_ECON = pd.read_csv(
    os.path.join(PATH3, 'agg_ntnl_mnthly.csv'), parse_dates=['DATE'])

# narrow down by yrgrp
DF_ECON.loc[:, 'DATE'] = DF_ECON.loc[:, 'DATE'].dt.to_period('m')
df.loc[:, 'PRD'] = df.loc[:, 'PRD'].dt.to_period('m')
df.loc[:, 'ORIG_DTE'] = df.loc[:, 'ORIG_DTE'].dt.to_period('m')

pd.period_range(a, b, freq='M')

# create year groups
yr_grps = [[2000], [2005, 2006, 2007, 2008],
           [2001, 2002, 2003, 2004],
           [2009, 2010, 2011, 2012, 2013, 2014, 2015]]

############################################################
# Time-independent models
############################################################
all_vins = df['ORIG_DTE'].unique()
valid_vins = all_vins[[True if v.year in yr_grps[1] else False
                       for v in all_vins]]
red_df = df[df.ORIG_DTE.isin(valid_vins)]

cat_vars = ['ORIG_CHN', 'PURPOSE']
cont_vars = ['ORIG_AMT', 'NUM_BO', 'DTI', 'CSCORE_MN', 'ORIG_VAL']
other = ['LOAN_ID', 'ORIG_DTE', 'did_dflt', 'PRD']

df_cox = utility.drop_NA_cols(red_df[other + cont_vars + cat_vars])
df_cox = df_cox.dropna(axis=0, how='any')

# create age
df_cox['AGE'] = (df_cox['PRD'] - df_cox['ORIG_DTE']).astype(int)

############################################################
# KaplanMeierFitter
############################################################
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

T = df_cox["AGE"]
E = df_cox["did_dflt"]

kmf.fit(T, event_observed=E)
kmf.plot()

# univariate analysis
# ORIG_CHN
ax = plt.subplot()
for chn in df_cox.ORIG_CHN.unique():
    is_chn = (df_cox.ORIG_CHN == chn)
    kmf.fit(T[is_chn], event_observed=E[is_chn], label=chn)
    kmf.plot(ax=ax)

# PURPOSE
ax = plt.subplot()
for purpose in df_cox.PURPOSE.unique():
    is_pur = (df_cox.PURPOSE == purpose)
    kmf.fit(T[is_pur], event_observed=E[is_pur], label=purpose)
    kmf.plot(ax=ax)

# Test kmf on test set
kmf = KaplanMeierFitter()
cph = CoxPHFitter()

# right censoring must be independent of dflt
# check if loans that didn't dflt all go to the last period
df_cox[df_cox.did_dflt == 0].PRD.value_counts()

############################################################
# NOTE: Cox is always underpredicting bc many in test set are censored
# Thus not had the chance to experience the full extent
############################################################
ts = int(df_cox.shape[0] * 0.1)
ts = 4
for train, test, trial_i in utility.train_test_splitter(df_cox, 5, ts, key='ORIG_DTE'):
    break

kmf.fit(train["AGE"], event_observed=train["did_dflt"])

test['year'] = test.ORIG_DTE.apply(lambda x: x.year)
year_cutoff = 2005
rtest = test[test.year <= year_cutoff]
rtest = test

T, E = rtest['AGE'], rtest['did_dflt']
did_die = E == 1

for time_cutoff in range(72, 132, 12):
    # remove loans censored before time_cutoff
    rT = T[(T > time_cutoff) | (did_die)]
    rE = E[(T > time_cutoff) | (did_die)]
    for_plt = pd.DataFrame({'rT': rT, 'rE': rE})
    kmf.plot()
    (1 - for_plt.groupby('rT')['rE'].sum().cumsum()/for_plt.shape[0])\
        .plot(label='actual')
    plt.xlim([0, time_cutoff+1])
    plt.axvline(x=time_cutoff, color='red', linestyle='--', lw=2, alpha=0.4)
    plt.legend()
    plt.title('{0:2f}% loans remain'.format(100*rT.shape[0]/T.shape[0]))
    plt.show()

# univariate analysis
# ORIG_CHN
ax = plt.subplot()
for chn in df_cox.ORIG_CHN.unique():
    is_chn = (df_cox.ORIG_CHN == chn)
    kmf.fit(T[is_chn], event_observed=E[is_chn], label=chn)
    kmf.plot(ax=ax)

# PURPOSE
ax = plt.subplot()
for purpose in df_cox.PURPOSE.unique():
    is_pur = (df_cox.PURPOSE == purpose)
    kmf.fit(T[is_pur], event_observed=E[is_pur], label=purpose)
    kmf.plot(ax=ax)

    
############################################################
# WeibullFitter
############################################################
from lifelines import WeibullFitter

wf = WeibullFitter()
wf.fit(T, E)
print(wf.lambda_, wf.rho_)
wf.print_summary()
wf.plot()

############################################################
# NelsonAalenFitter
############################################################
from lifelines import NelsonAalenFitter

naf = NelsonAalenFitter()

naf.fit(T, event_observed=E)
naf.plot()

# univariate analysis: cum hazard
# ORIG_CHN
ax = plt.subplot()
for chn in df_cox.ORIG_CHN.unique():
    is_chn = (df_cox.ORIG_CHN == chn)
    naf.fit(T[is_chn], event_observed=E[is_chn], label=chn)
    naf.plot(ax=ax)

# PURPOSE
ax = plt.subplot()
for purpose in df_cox.PURPOSE.unique():
    is_pur = (df_cox.PURPOSE == purpose)
    naf.fit(T[is_pur], event_observed=E[is_pur], label=purpose)
    naf.plot(ax=ax)

# univariate analysis: hazard fxn
# NOTE: no real distinction
b = 3
# ORIG_CHN
ax = plt.subplot()
for chn in df_cox.ORIG_CHN.unique():
    is_chn = (df_cox.ORIG_CHN == chn)
    naf.fit(T[is_chn], event_observed=E[is_chn], label=chn)
    naf.plot_hazard(ax=ax, bandwidth=b)

# PURPOSE
ax = plt.subplot()
for purpose in df_cox.PURPOSE.unique():
    is_pur = (df_cox.PURPOSE == purpose)
    naf.fit(T[is_pur], event_observed=E[is_pur], label=purpose)
    naf.plot_hazard(ax=ax, bandwidth=b)    


############################################################
# Cox proportional: statsmodels
############################################################
formula = ('AGE ~ {0} + {1}'
           .format('+'.join(['ORIG_CHN', 'PURPOSE'] +
                            ['NUM_BO', 'CSCORE_MN']),
                   '+'.join(['np.log({0})'.format(c)
                             for c in ['ORIG_AMT', 'ORIG_VAL']])))

# is cph appropriate? 
# 1. Checking the proportional hazards assumption
# plot the logs curve: the loglogs (-log(survival curve)) vs log(time). If the curves are parallel (and hence do not cross each other), then it’s likely the variable satisfies proportional hzrd assumption. If curves do cross, likely must “stratify” the variable
ts = int(df_cox.shape[0] * 0.1)
ts = 4
for train, test, trial_i in utility.train_test_splitter(df_cox, 5, ts, key='ORIG_DTE'):
    break

T, E = train['AGE'], train['did_dflt']
# ORIG_CHN
ax = plt.subplot()
for chn in train.ORIG_CHN.unique():
    is_chn = (df_cox.ORIG_CHN == chn)
    kmf.fit(T[is_chn], event_observed=E[is_chn], label=chn)
    kmf.plot_loglogs(ax=ax)

# PURPOSE
ax = plt.subplot()
for purpose in train.PURPOSE.unique():
    is_pur = (train.PURPOSE == purpose)
    kmf.fit(T[is_pur], event_observed=E[is_pur], label=purpose)
    kmf.plot_loglogs(ax=ax)

# fit cph
model = models.PHR(train, test, formula, 'did_dflt', 'LOAN_ID')
model.fit_model()

# 2. compare baseline of cph with kmf to see significance of covs
ax = plt.subplot()
kmf.fit(train['AGE'],
        event_observed=train['did_dflt'])
kmf.plot(ax=ax)
hz, base = model.make_pred(use_train=True)
ages = np.arange(train['AGE'].min(), train['AGE'].max() + 1)
baseline_surv = np.exp(-base(ages))
plt.scatter(x=ages, y=baseline_surv, s=0.5)
plt.legend()

# Test cph on test set
test_pred = model.make_pred()

T, E = test['AGE'], test['did_dflt']
did_die = E == 1
uniq_ids = test['LOAN_ID']

for time_cutoff in range(72, 132, 12):
    cond = (T > time_cutoff) | (did_die)
    
    # remove loans censored before time_cutoff from actual data
    for_plt = pd.DataFrame({'rT': T[cond], 'rE': E[cond]})
    (1 - for_plt.groupby('rT')['rE'].sum().cumsum()/for_plt.shape[0])\
        .plot(label='actual')

    # remove laons censored before time_cutoff from test_pred
    rtest_pred = test_pred[test_pred.ID.isin(uniq_ids[cond])]
    (1 - rtest_pred.groupby('AGE')['cum_death_preds'].sum()/rtest_pred.shape[0])\
        .plot(label='predicted')

    plt.xlim([0, time_cutoff+1])
    plt.axvline(x=time_cutoff, color='red', linestyle='--', lw=2, alpha=0.4)
    plt.legend()
    plt.title('{0:2f}% loans remain'.format(100*rT.shape[0]/T.shape[0]))
    plt.show()
