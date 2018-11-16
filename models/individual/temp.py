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
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import statsmodels.formula.api as smf
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

############################################################
# statsmodels genericlikelihoodmodel
############################################################
data = sm.datasets.spector.load_pandas()
exog = data.exog
endog = data.endog
print(sm.datasets.spector.NOTE)
print(data.exog.head())

exog = sm.add_constant(exog, prepend=True)


class MyProbit(GenericLikelihoodModel):
    def loglike(self, params):
        exog = self.exog
        endog = self.endog
        q = 2 * endog - 1
        return stats.norm.logcdf(q*np.dot(exog, params)).sum()

sm_probit_manual = MyProbit(endog, exog).fit()
print(sm_probit_manual.summary())

sm_probit_canned = sm.Probit(endog, exog).fit()
print(sm_probit_canned.params)
print(sm_probit_manual.params)
print(sm_probit_canned.cov_params())
print(sm_probit_manual.cov_params())

############################################################
# Cox-proportional
############################################################
data = sm.datasets.get_rdataset("flchain", "survival").data
del data["chapter"]
data = data.dropna()
data["lam"] = data["lambda"]
data["female"] = (data["sex"] == "F").astype(int)
data["year"] = data["sample.yr"] - min(data["sample.yr"])
status = data["death"].values

mod = smf.phreg("futime ~ age + female + creatinine + "
                "np.sqrt(kappa) + np.sqrt(lam) + year + mgus",
                data, status=status, ties="efron")
rslt = mod.fit()
print(rslt.summary())

############################################################
# phreg on loan data
############################################################
PATH1 = '~/Google Drive/PWBM/CECL/data'
# PATH1 = 'c:/Users/hanjh/Documents/Google Drive/PWBM/CECL/data/fannie_mae'
PATH2 = os.path.join(PATH1, 'fannie_mae', 'clean')
filename = 'noMOD_5M'
YEARS = ['2000Q4', '2001Q4', '2002Q4', '2003Q4', '2004Q4', '2005Q4']

df = pd.read_csv(
    os.path.join(PATH2, 'last_row_only_{1}_{0}.csv'.format(
        filename, '.'.join(YEARS))),
    parse_dates=['PRD', 'ORIG_DTE'])

# delete unnecessary columns
del df['Unnamed: 0'], df['I']
df.drop(df.loc[:1, 'LIBOR1':'"Far West"'].columns, axis=1, inplace=True)
df.drop(df.loc[:1, 'dflt_in1yr':'dflt_geq9yr'].columns, axis=1, inplace=True)

# change date format
df.loc[:, 'PRD'] = df.loc[:, 'PRD'].dt.to_period('m')
df.loc[:, 'ORIG_DTE'] = df.loc[:, 'ORIG_DTE'].dt.to_period('m')

# remove prepaid loans
df = df[df['LAST_STAT'] != 'P']

# show last statuses of dflt/non-dflt loans
df.loc[df['did_dflt'] == 0, 'LAST_STAT'].value_counts()
df.loc[df['did_dflt'] == 1, 'LAST_STAT'].value_counts()

# did_dflt is 'status'
# AGE is 'endog'
# exog must be available at loan origination
exog = ['ORIG_CHN', 'ORIG_RT', 'np.log(ORIG_AMT)', 'OCLTV', 'NUM_BO', 'DTI', 'FTHB_FLG',
        'PURPOSE', 'PROP_TYP', 'NUM_UNIT', 'OCC_STAT', 'CSCORE_MN']

formula = 'AGE ~ {0}'.format('+'.join(exog))

# get variable names
regex = re.compile(r'\)$|^np.log\(|^np.sqrt\(')
var_names = [re.sub(regex, '', e) for e in exog] + ['AGE'] + ['did_dflt']
df_reg = df[var_names]
# check NAs
df_reg.apply(lambda x: x.isna().mean(), axis=0)
# remove NAs
df_reg.dropna(axis=0, how='any', inplace=True)

fit = smf.phreg(formula, df_reg, status=df_reg['did_dflt'], ties="efron").fit()
fit.summary()

# get hazard ratio predictions
hz_ratios = fit.predict(pred_type='hr').predicted_values

# get cumulative hazard
def cum_surv(age_range, hz_ratios):
    # age_range: range of age values to forecast
    # hz_ratios: hazard ratio of all observations
    # returns matrix
    base_cum_hz_fxn = fit.baseline_cumulative_hazard_function[0]
    return np.exp(-base_cum_hz_fxn(age_range).reshape(-1,1) * hz_ratios)

# pred vs default for num_loans
age_range = np.arange(1, 250)
cum_death = (1 - cum_surv(age_range, hz_ratios)).sum(axis=1)
pred_for_plt = pd.DataFrame({'Predicted':cum_death}, index=age_range)
pred_for_plt.index.name = 'AGE'

for_plt = df_reg.sort_values('AGE')[['AGE', 'did_dflt']].rename(index=str, columns={'did_dflt':'Actual'})
dflt_pct = for_plt.groupby('AGE').sum()
cum_sum = dflt_pct.cumsum()

ax = plt.subplot()
cum_sum.plot(ax=ax)
pred_for_plt.plot(ax=ax)
plt.show()

# pred vs default for num_loans
age_range = np.arange(1, 250)
cum_death_per_loan = 1 - cum_surv(age_range, hz_ratios)

pred_for_plt = pd.DataFrame({'Predicted':cum_death}, index=age_range)
pred_for_plt.index.name = 'AGE'

for_plt = df_reg.sort_values('AGE')[['AGE', 'did_dflt']].rename(index=str, columns={'did_dflt':'Actual'})
dflt_pct = for_plt.groupby('AGE').sum()
cum_sum = dflt_pct.cumsum()

ax = plt.subplot()
cum_sum.plot(ax=ax)
pred_for_plt.plot(ax=ax)
plt.show()


############################################################
# lifelines
############################################################
rossi_dataset = load_rossi()
cph = CoxPHFitter()
cph.fit(rossi_dataset, duration_col='week', event_col='arrest', show_progress=True)
cph.print_summary()

# look at coefs
cph.plot()
plt.show()

# compare with KMF
kmf = KaplanMeierFitter()
kmf.fit(rossi_dataset["week"], event_observed=rossi_dataset["arrest"])
# plot both
ax = plt.subplot()
cph.baseline_survival_.plot(ax=ax)
kmf.survival_function_.plot(ax=ax)
plt.show()

############################################################
# lifelines on loan data
############################################################
PATH1 = '~/Google Drive/PWBM/CECL/data'
# PATH1 = 'c:/Users/hanjh/Documents/Google Drive/PWBM/CECL/data/fannie_mae'
PATH2 = os.path.join(PATH1, 'fannie_mae', 'clean')
filename = 'noMOD_5M'
YEARS = ['2000Q4', '2001Q4', '2002Q4', '2003Q4', '2004Q4', '2005Q4']

df = pd.read_csv(
    os.path.join(PATH2, 'last_row_only_{1}_{0}.csv'.format(
        filename, '.'.join(YEARS))),
    parse_dates=['PRD', 'ORIG_DTE'])

# delete unnecessary columns
del df['Unnamed: 0'], df['I']
df.drop(df.loc[:1, 'LIBOR1':'"Far West"'].columns, axis=1, inplace=True)
df.drop(df.loc[:1, 'dflt_in1yr':'dflt_geq9yr'].columns, axis=1, inplace=True)

# change date format
df.loc[:, 'PRD'] = df.loc[:, 'PRD'].dt.to_period('m')
df.loc[:, 'ORIG_DTE'] = df.loc[:, 'ORIG_DTE'].dt.to_period('m')

# remove prepaid loans
df = df[df['LAST_STAT'] != 'P']

# show last statuses of dflt/non-dflt loans
df.loc[df['did_dflt'] == 0, 'LAST_STAT'].value_counts()
df.loc[df['did_dflt'] == 1, 'LAST_STAT'].value_counts()

# did_dflt is 'status'
# AGE is 'endog'
# exog must be available at loan origination
exog_cat = ['ORIG_CHN', 'FTHB_FLG', 'PURPOSE', 'PROP_TYP', 'OCC_STAT']
exog_cont = ['ORIG_RT', 'ORIG_AMT', 'OCLTV', 'NUM_BO', 'DTI', 'NUM_UNIT', 'CSCORE_MN']
var_names = exog_cat + exog_cont + ['AGE'] + ['did_dflt']
df_reg = df[var_names]
# check NAs
df_reg.apply(lambda x: x.isna().mean(), axis=0)
# remove NAs
df_reg.dropna(axis=0, how='any', inplace=True)
cat = exog_cat[0]
for cat in exog_cat:
    dummies = pd.get_dummies(df_reg[cat], prefix=cat, drop_first=True)
    df_reg = pd.concat([df_reg, dummies], axis=1, join='outer')
    del df_reg[cat]

cph = CoxPHFitter()
cph.fit(df_reg, duration_col='AGE', event_col='did_dflt', show_progress=True)
cph.print_summary()

# look at coefs
cph.plot()
plt.show()

# compare with KMF
kmf = KaplanMeierFitter()
kmf.fit(df_reg['AGE'], event_observed=df_reg['did_dflt'])
# plot both
ax = plt.subplot()
cph.baseline_survival_.plot(ax=ax)
kmf.survival_function_.plot(ax=ax)
plt.show()
