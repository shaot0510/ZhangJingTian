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
from numba import jit
# import line_profiler

from scipy.interpolate import interp1d

import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.base.model import GenericLikelihoodModel
import statsmodels.formula.api as smf

from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter

from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics.scorer import make_scorer
from sklearn.utils import resample

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

PATH = os.path.join('fannie_mae', 'individual')
PATH1 = 'data'
PATH2 = os.path.join(PATH1, 'fannie_mae', 'clean')
PATH3 = os.path.join(PATH1, 'economic')

filetype = 'indv_500K'
YEARS = ['{0}Q{1}'.format(y, q) for y in range(2000, 2016) for q in [4]]
filename = '.'.join([YEARS[0], YEARS[-1], 'inc3'])
# filename = '2000Q1'


# gets roc_auc and plots if plotit=True
def get_roc(true, score, plotit=True):
    fpr, tpr, _ = roc_curve(y_true=true, y_score=score)
    roc_auc = auc(fpr, tpr)
    if plotit:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc


def get_best_n(base_model, base_model_kwargs, gkf_groups,
               df, formula, error_type, plotit=True):
    # Run classifier with cross-validation and plot ROC curves
    gkf = GroupKFold(n_splits=5)
    mod_name = base_model.__name__
    print('Cross validating for {0} with formula: {1}'.format(mod_name,
                                                              formula))
    # sizes of GBM to try out
    n_ests = (list(range(5, 40, 10)) + list(range(40, 60, 5)) +
              list(range(60, 90, 10)))
    model_kwargs = base_model_kwargs.copy()
    for n in n_ests:
        if n == n_ests[0]:
            mean_score, std_score = [], []
        print('\n'+('=' * 50))
        print('Processing n = ' + str(n))
        model_kwargs['n_estimators'] = n
        scores = []
        for cv_train_index, cv_test_index in gkf.split(df, groups=gkf_groups):
            fit = fit_model(base_model,
                            model_kwargs,
                            df.iloc[cv_train_index],
                            formula)
            preds = get_preds(base_model, fit,
                              df.iloc[cv_test_index], formula)
            y_test, X_test = patsy.dmatrices(formula,
                                             df.iloc[cv_test_index],
                                             return_type='dataframe')
            if error_type == 'score':
                # Compute R^2 with given test data
                scores.append(fit.score(X_test, y_test))
            elif error_type == 'MSE':
                # Compute MSE with given test data
                se = (preds - y_test)**2
                scores.append(-se.sum())
            elif error_type == 'ROC':
                # Compute ROC curve and area the curve
                scores.append(roc_auc_score(y_test, preds))
            else:
                print('{0} not yet implemented'.format(error_type))
        mean_score.append(np.array(scores).mean())
        std_score.append(np.array(scores).std())

    argmax_ = np.array(mean_score).argmax()
    max_mean_auc, best_n = mean_score[argmax_], n_ests[argmax_]

    if plotit:
        plt.errorbar(n_ests, mean_score, yerr=std_score, fmt='o')
        plt.axvline(best_n, linestyle='--', lw=0.4, color="r",
                    alpha=.8, label=('Best n: {0} with '
                                     'auc {1:.3f}').format(best_n,
                                                           max_mean_auc))
        plt.legend()
        plt.show()
    return best_n


def prepare_data(orig_df, do_prd_expand=False,
                 is_bootstrap=False, full_df=None):
    res = get_sample(orig_df.copy(), do_prd_expand, is_bootstrap, full_df)
    if do_prd_expand:
        # attach econ vars
        res = res.merge(
            DF_ECON, how='left', left_on='PRD', right_on='DATE', copy=False)
        del res['DATE']
        ############################################################
        # ISSUE: test might contain columns dropped in train
        ############################################################
        # res = drop_NA_cols(res)
        res.dropna(axis=0, how='any')

    # change date format; not done beforehand
    # because of prd_expansion (date_range)
    res.loc[:, 'PRD'] = res.loc[:, 'PRD'].dt.to_period('m')
    res.loc[:, 'ORIG_DTE'] = res.loc[:, 'ORIG_DTE'].dt.to_period('m')
    # calculate AGE
    res['AGE'] = (res['PRD'] - res['ORIG_DTE']).astype('int')
    return res


def convert_dummies(df, cat_vars):
    print('Converting categorical to dummies')
    # convert cat_vars to dummies
    to_concat = patsy.dmatrix(' + '.join(cat_vars),
                              df, return_type='dataframe')
    # rename columns and remove intercept
    regex = re.compile('\[T\.|\]$')
    to_concat.columns = [regex.sub('_', t) for t in to_concat.columns.tolist()]
    # delete Intercept
    del to_concat['Intercept']
    # drop cat_vars
    # concat cat_vars
    return pd.concat([df.drop([v for v in df.columns if v in cat_vars],
                              axis=1), to_concat], axis=1)


# def f1():
#     from collections import Counter

#     ids = full_df['LOAN_ID'].unique()
#     select_ids = np.random.choice(ids, len(ids), replace=True)
#     temp = full_df.set_index('LOAN_ID')

#     to_concats = []
#     for i in select_ids:
#         to_concats.append(temp.loc[i])
#     return pd.concat(to_concats, axis=0)

# def f2():
#     df = resample(train)
#     return get_PRD_expansion(df)


def get_sample(df, do_prd_expand, is_bootstrap, full_df=None):
    # if bootstrapping, draw bootstrap sample
    # full_df is the full dataset that's prd_expanded
    if is_bootstrap:
        print('Bootstrapping data')
        df = resample(df)
    if do_prd_expand:
        print('Performing period expansion')
        # expand by PRD
        return get_PRD_expansion(df)
    else:
        return df


def fit_model(base_model, base_model_kwargs, df, formula):
    mod_name = base_model.__name__
    model_kwargs = base_model_kwargs.copy()
    print('Fitting {0} with formula: {1}'.format(mod_name, formula))
    # COX
    if mod_name == 'PHReg':
        (model_kwargs['endog'],
         model_kwargs['exog']) = patsy.dmatrices(formula, df,
                                                 return_type='dataframe')
        model_kwargs['status'] = df['did_dflt']
        return base_model(**model_kwargs).fit()
    # COX time
    elif mod_name == 'CoxTimeVaryingFitter':
        new_formula = formula.split('~')[1]
        new_formula = ' + '.join([new_formula, 'AGE', 'LOAN_ID', 'did_dflt'])
        temp = patsy.dmatrix(new_formula, df, return_type='dataframe')
        temp['AGE+1'] = temp['AGE'] + 1
        (model_kwargs['df'],
         model_kwargs['start_col'],
         model_kwargs['id_col'],
         model_kwargs['event_col'],
         model_kwargs['stop_col']) = [temp, 'AGE', 'LOAN_ID', 'did_dflt', 'AGE+1']

        return base_model().fit(**model_kwargs)
    # GBM
    elif mod_name == 'GradientBoostingClassifier':
        y, X = patsy.dmatrices(formula, df, return_type='dataframe')
        return base_model(**model_kwargs).fit(X, y.values.ravel())
    # RF
    elif mod_name == 'RandomForestClassifier':
        y, X = patsy.dmatrices(formula, df, return_type='dataframe')
        return base_model(**model_kwargs).fit(X, y.values.ravel())
    else:
        print('{0} not yet implemented'.format(mod_name))


def get_preds(base_model, fit, new_df, formula):
    '''preds is the probability of event occuring at each age'''
    # P(t < T <= t+1) = F(t+1) - F(t) and P(T <= 0) = 0
    def get_diff_by_ID(x):
        v = x.values
        if len(v) >= 2:
            return np.append(0, v[1:] - v[:-1])
        else:
            return v

    mod_name = base_model.__name__
    print('Making predictions for {0} '
          'with formula: {1}'.format(mod_name, formula))
    pred_kwargs = {}

    # COX
    if mod_name == 'PHReg':
        ages, pred_kwargs['exog'] = patsy.dmatrices(formula,
                                                    new_df,
                                                    return_type='dataframe')
        pred_kwargs['pred_type'] = 'hr'
        hz_ratios = fit.predict(**pred_kwargs).predicted_values
        base_cum_hz_fxn = fit.baseline_cumulative_hazard_function[0]
        cum_death_preds = (1-np.exp(-base_cum_hz_fxn(ages).ravel() *
                                    hz_ratios))
        temp = pd.DataFrame({'ID': new_df['LOAN_ID'],
                             'cum_death_preds': cum_death_preds})
        preds = temp.groupby('ID').transform(get_diff_by_ID).values
    # RF & GBM
    elif mod_name in ['GradientBoostingClassifier', 'RandomForestClassifier']:
        _, pred_kwargs['X'] = patsy.dmatrices(formula,
                                              new_df,
                                              return_type='dataframe')
        preds = fit.predict_proba(**pred_kwargs)[:, 1]
    elif mod_name == 'CoxTimeVaryingFitter':
        ages, pred_kwargs['X'] = patsy.dmatrices(formula,
                                                 new_df,
                                                 return_type='dataframe')
        hz_ratios = fit.predict_partial_hazard(**pred_kwargs).values
        base_cum_hz = fit.baseline_cumulative_hazard_

        # append np.inf at ends for extrapolation
        # same implementation as statsmodels phreg
        xp, fp = base_cum_hz.index, base_cum_hz.values.ravel()
        xp = np.r_[-np.inf, xp, np.inf]
        fp = np.r_[fp[0], fp, fp[-1]]
        base_cum_hz_fxn = interp1d(xp, fp, kind='zero')

        cum_death_preds = 1-np.exp(-base_cum_hz_fxn(ages).ravel() *
                                   hz_ratios.ravel())
        temp = pd.DataFrame({'ID': new_df['LOAN_ID'],
                             'cum_death_preds': cum_death_preds})
        preds = temp.groupby('ID').transform(get_diff_by_ID).values
    else:
        print('{0} not yet implemented'.format(mod_name))
        return
    return preds.ravel()


def get_btstrp(base_model, base_model_kwargs, test, train,
               formula, do_prd_expand, btstrp_trials=30, lo=2.5):
    '''returns df (test.shape[0], 2) that is '''
    '''the lo, hi percentile btstrp prediction'''
    ############################################################
    # FIX: BOOTSTRP samples must preserve entire period of loan
    ############################################################
    test_data = prepare_data(test, do_prd_expand=True)
    # train_data = prepare_data(train, do_prd_expand=True)
    hi = 100 - lo
    # initial df
    btstrp_all = pd.DataFrame()
    for btstrp_trial_i in range(btstrp_trials):
        print('\nBootstrp trial {0}'.format(btstrp_trial_i))
        btstrp_train = prepare_data(train, do_prd_expand=do_prd_expand,
                                    is_bootstrap=True)
        fit = fit_model(base_model, base_model_kwargs, btstrp_train, formula)
        preds = get_preds(base_model, fit, test_data, formula)
        to_concat = pd.DataFrame({'btstrp_pred': preds}, index=test_data.index)
        if btstrp_all.shape[0] == 0:
            btstrp_all = to_concat
        else:
            btstrp_all = pd.concat([btstrp_all, to_concat], axis=1)

    print('\nCompiling bootstrap data')
    # get lo, hi percentiles by row and split the tuple into 2 column df
    btstrp_rslt = btstrp_all.apply(lambda x: (np.percentile(x, lo),
                                              np.percentile(x, hi)),
                                   axis=1).apply(pd.Series)
    btstrp_rslt.columns = ['{0}%'.format(lo), '{0}%'.format(hi)]
    return btstrp_rslt


def plot_btstrp(for_plt, x, y, pred_y, lo_name, hi_name, ax=None):
    print('Generating plot')
    if ax is None:
        ax = plt.subplot()
    sns.lineplot(x=x, y=lo_name, data=for_plt, linewidth=0.5,
                 ax=ax, color='#75bbfd', linestyle='--')
    sns.lineplot(x=x, y=hi_name, data=for_plt, linewidth=0.5,
                 ax=ax, color='#75bbfd', linestyle='--', label='CI')
    l1, l2 = ax.lines[0], ax.lines[1]

    # Get the xy data from the lines so that we can shade
    d1, d2 = l1.get_xydata(), l2.get_xydata()
    ax.fill_between(d1[:, 0], y1=d1[:, 1], y2=d2[:, 1],
                    color="#95d0fc", alpha=0.3)
    sns.lineplot(x=x, y=pred_y, data=for_plt,
                 ax=ax, linewidth=1.5, label='Prediction', color='#0165fc')
    sns.lineplot(x=x, y=y, data=for_plt,
                 ax=ax, linewidth=1.5, label='Actual', color='#c04e01')
    # ax.set_title('Cumulative Prediction [{0}, {1}]'.format(lo_name, hi_name))


def drop_NA_cols(df, cutoff=0.01, excluded=['CSCORE_MN']):
    # check NAs
    col_na = df.apply(lambda x: x.isna().mean(), axis=0)
    drop_cols = col_na[col_na > cutoff]
    print('Drop candidates: \n{0}'.format(drop_cols))
    print('Excluded: {0}'.format(', '.join(excluded)))
    dropped = [v for v in drop_cols.index if v not in excluded]
    print('Dropped: {0}'.format(', '.join(dropped)))
    return df.drop(dropped, axis=1)


def split_train_test(df, key, test_size=None, p=None):
    # divide train/test
    vins = df[key].unique()
    if p is None:
        p = np.ones(len(vins))
    if test_size:
        test_vins = np.random.choice(vins, test_size,
                                     replace=False, p=p/sum(p))
    else:
        test_vins = np.random.choice(vins, 1, replace=False, p=p/sum(p))
        counts = df[key].value_counts()
        temp_p = p.copy()
        while counts[test_vins].sum() < 1000:
            test_vins = np.append(test_vins,
                                  np.random.choice(vins, 1,
                                                   replace=False,
                                                   p=temp_p/sum(temp_p)))
            is_zero = np.array([v in test_vins for v in vins])
            temp_p[is_zero] = 0
    print('Test set: {0}'.format(', '.join([str(v) for v in test_vins])))
    is_zero = np.array([v in test_vins for v in vins])
    p[is_zero] = 0
    train_vins = [v for v in vins if v not in test_vins]
    test, train = df[df[key].isin(test_vins)], df[df[key].isin(train_vins)]
    print('Test size: {0}, Train size: {1}'.format(test.shape[0],
                                                   train.shape[0]))
    return test, train, p


def get_PRD_expansion(orig_df):
    '''create full length dataframe with period dates in order to add macros'''
    '''ORIG_DTE: start date, PRD: last recorded date of loan'''
    # only get unique loans
    df_dates = orig_df[['LOAN_ID', 'ORIG_DTE', 'PRD']].set_index('LOAN_ID')
    min_date, max_date = df_dates['ORIG_DTE'].min(), df_dates['PRD'].max()
    keys = pd.date_range(min_date, max_date, freq='MS').values
    values = np.arange(len(keys))
    date2ind = dict(zip(keys, values))
    ind2date = dict(zip(values, keys))

    loan_ids_l, prd_l = [], []
    for loan_id in df_dates.index:
        vals = df_dates.loc[loan_id].values
        if isinstance(vals[0], np.ndarray):
            size = vals.shape[0]
            start, end = vals[0]
        else:
            size = 1
            start, end = vals
        inds_to_append = list(range(date2ind[start], date2ind[end]+1))
        prd_l += inds_to_append * size
        loan_ids_l += ([loan_id] * len(inds_to_append)) * size

    temp = pd.DataFrame({
        'LOAN_ID': loan_ids_l,
        'PRD_I': prd_l}, dtype=np.int64).set_index('LOAN_ID')
    temp['PRD'] = temp['PRD_I'].apply(lambda x: ind2date[x])
    del temp['PRD_I']

    # with PRD in var_names PRD_x is current PRD, PRD_y is last PRD for loan
    temp = temp.merge(
        orig_df.drop_duplicates('LOAN_ID').set_index('LOAN_ID'),
        how='left',
        left_index=True,
        right_index=True).reset_index()

    # rename did_dflt so only applies to last row of each loan
    temp['is_later'] = 1*(temp['PRD_x'] == temp['PRD_y'])
    temp['did_dflt'] = temp['is_later'] * temp['did_dflt']

    # remove extra PRD
    del temp['PRD_y'], temp['is_later'],
    return temp.rename(index=str, columns={'PRD_x': 'PRD'})


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
# drop unnecessary cols
DF_ECON.drop(['IR', 'LIBOR', 'UNEMP'], axis=1, inplace=True)
# convert col to ensure high variance
for c in DF_ECON:
    if c != 'DATE':
        DF_ECON[c] = 100 * DF_ECON[c]

############################################################
# Compare
############################################################
cat_vars = ['ORIG_CHN', 'PURPOSE']
cont_vars = ['ORIG_AMT', 'NUM_BO', 'DTI', 'CSCORE_MN', 'ORIG_VAL']
other = ['LOAN_ID', 'ORIG_DTE', 'did_dflt', 'PRD']
must_log = ['ORIG_AMT', 'ORIG_VAL', 'ORIG_RT']

df_cox = drop_NA_cols(df[other + cont_vars + cat_vars], True)
df_cox = df_cox.dropna(axis=0, how='any')

df_cox = convert_dummies(df_cox, cat_vars)

p = np.ones(len(df_cox['ORIG_DTE'].unique()))

for trial_i in range(20):
    test, train, p = split_train_test(df_cox, 'ORIG_DTE', p=p)

    ############################################################
    # Cox
    ############################################################
    train_cox = prepare_data(train, do_prd_expand=False)

    base_model = PHReg
    base_model_kwargs = {'ties': 'efron'}

    # not including intercept causes first cat_var to not have base case
    # with intercept gives same result
    modified_vars = []
    for v in train_cox.columns:
        if v not in other and v != 'AGE':
            if v in must_log:
                modified_vars.append('np.log({0})'.format(v))
            else:
                modified_vars.append(v)

    # remove intercept
    formula = 'AGE ~ -1 + {0}'.format(' + '.join(modified_vars))

    fit1 = fit_model(base_model, base_model_kwargs, train_cox, formula)
    print(fit1.summary())

    ############################################################
    # Train fit
    ############################################################
    train_cox_P = prepare_data(train, do_prd_expand=True)
    train_coxt = train_cox_P.copy()

    # # drop all macros
    # train_cox_P.drop(['PRD', 'ORIG_DTE', 'LIBOR', 'CPI',
    #                   'UNEMP', 'MR', 'HPI', 'rGDP'],
    #                  axis=1, inplace=True)

    # drop NAs
    a = train_cox_P.shape
    train_cox_P = train_cox_P.dropna(axis=0, how='any')
    print('{0} -> {1}'.format(a, train_cox_P.shape))

    train_cox_P['pred_probs'] = get_preds(base_model, fit1,
                                          train_cox_P, formula)
    # ROC results
    train_roc_auc = get_roc(train_cox_P['did_dflt'],
                            train_cox_P['pred_probs'], plotit=False)
    print('Train auc: {0:.4f}'.format(train_roc_auc))

    # projections
    x, y, pred_y = 'AGE', 'did_dflt', 'pred_probs'
    for_plt = train_cox_P.groupby('AGE')[[y, pred_y]]\
                         .sum().cumsum().reset_index()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    sns.lineplot(x=x, y=pred_y, data=for_plt, linewidth=1.5,
                 label='Prediction', color='#0165fc')
    sns.lineplot(x=x, y=y, data=for_plt, linewidth=1.5,
                 label='Actual', color='#c04e01')

    ############################################################
    # Test fit
    ############################################################
    # expand test set as PRD_range
    # var_names can't include vars that vary with PRD (ie. AGE)
    # did_dflt does vary with PRD but we redefine it
    # PRD here represents last PRD so doesn't vary with PRD
    test_cox = prepare_data(test, do_prd_expand=True)
    test_coxt = test_cox.copy()

    test_cox['pred_probs'] = get_preds(base_model, fit1, test_cox, formula)

    # ROC results
    test_roc_auc = get_roc(test_cox['did_dflt'],
                           test_cox['pred_probs'], plotit=False)
    print('Test auc: {0:.4f}'.format(test_roc_auc))

    # bootstraps
    print('Running bootstraps')
    # note we have to use df in indv format
    btstrp_int = get_btstrp(base_model, base_model_kwargs, test, train,
                            formula, do_prd_expand=False)
    lo_name, hi_name = btstrp_int.columns
    btstrp_df = pd.concat([test_cox, btstrp_int], axis=1)
    x, y, pred_y = 'AGE', 'did_dflt', 'pred_probs'
    for_plt = btstrp_df.groupby('AGE')[[y, pred_y, lo_name, hi_name]]\
                       .sum().cumsum().reset_index()
    plot_btstrp(for_plt, x, y, pred_y, lo_name, hi_name,
                ax=plt.subplot(2, 2, 2))

    ############################################################
    # Cox Time-varying
    ############################################################
    # drop unnecessary cols
    train_coxt.drop(['PRD', 'ORIG_DTE'],
                    axis=1, inplace=True)

    # fit model
    base_model = CoxTimeVaryingFitter
    base_model_kwargs = {'show_progress': True}

    # not including intercept causes first cat_var to not have base case
    # with intercept gives same result
    modified_vars = []
    for v in train_coxt.columns:
        if v not in other + ['AGE', 'pred_probs']:
            if v in must_log:
                modified_vars.append('np.log({0})'.format(v))
            else:
                modified_vars.append(v)

    # remove intercept
    formula = 'AGE ~ -1 + {0}'.format(' + '.join(modified_vars))

    fit1 = fit_model(base_model, base_model_kwargs, train_coxt, formula)
    fit1.print_summary()

    ############################################################
    # Train fit
    ############################################################
    train_coxt['pred_probs'] = get_preds(base_model, fit1, train_coxt, formula)

    # ROC results
    train_roc_auc = get_roc(train_coxt['did_dflt'],
                            train_coxt['pred_probs'], plotit=False)
    print('Train auc: {0:.4f}'.format(train_roc_auc))

    # projections
    x, y, pred_y = 'AGE', 'did_dflt', 'pred_probs'
    for_plt = train_coxt.groupby('AGE')[[y, pred_y]]\
                        .sum().cumsum().reset_index()

    plt.subplot(2, 2, 3)
    sns.lineplot(x=x, y=pred_y, data=for_plt, linewidth=1.5,
                 label='Prediction', color='#0165fc')
    sns.lineplot(x=x, y=y, data=for_plt, linewidth=1.5,
                 label='Actual', color='#c04e01')

    ############################################################
    # Test fit
    ############################################################
    # drop unnecessary cols
    test_coxt.drop(['PRD', 'ORIG_DTE'], axis=1, inplace=True)

    # drop NAs
    a = test_coxt.shape
    test_coxt = test_coxt.dropna(axis=0, how='any')
    print('{0} -> {1}'.format(a, test_coxt.shape))

    test_coxt['pred_probs'] = get_preds(base_model, fit1, test_coxt, formula)

    # ROC results
    test_roc_auc = get_roc(test_coxt['did_dflt'],
                           test_coxt['pred_probs'], plotit=False)
    print('Test auc: {0:.4f}'.format(test_roc_auc))

    # # bootstraps
    # print('Running bootstraps')
    # # note we have to use df in indv format
    # btstrp_int = get_btstrp(base_model, base_model_kwargs, test, train,
    #                         formula, do_prd_expand=True)
    # lo_name, hi_name = btstrp_int.columns
    # btstrp_df = pd.concat([test_coxt, btstrp_int], axis=1)
    # x, y, pred_y = 'AGE', 'did_dflt', 'pred_probs'
    # for_plt = btstrp_df.groupby('AGE')[[y, pred_y, lo_name, hi_name]]\
    #                    .sum().cumsum().reset_index()
    # plot_btstrp(for_plt, x, y, pred_y, lo_name, hi_name,
    #             ax=plt.subplot(2, 2, 4))

    # projections
    x, y, pred_y = 'AGE', 'did_dflt', 'pred_probs'
    for_plt = test_coxt.groupby('AGE')[[y, pred_y]]\
                       .sum().cumsum().reset_index()

    plt.subplot(2, 2, 4)
    sns.lineplot(x=x, y=pred_y, data=for_plt, linewidth=1.5,
                 label='Prediction', color='#0165fc')
    sns.lineplot(x=x, y=y, data=for_plt, linewidth=1.5,
                 label='Actual', color='#c04e01')

    # plot settings
    test_dates = '.'.join(pd.Series(test.ORIG_DTE.unique())
                          .dt.to_period('M').astype('str').tolist()[:5])
    test_count = test.shape[0]
    plt.suptitle('{0} loans from {1}+{2}'.format(test_count, test_dates,
                                                 max(0, len(test_dates)-5)),
                 fontsize=12)
    # plt.tight_layout()
    plt.legend()
    # plt.show()
    plotname = '.'.join([str(trial_i), 'COX_cum_sum_comparison',
                         filename, 'eps'])
    plt.savefig(os.path.join(PATH, 'plots', plotname), dpi=300,
                format='eps', bbox_inches='tight')
    plt.clf()
