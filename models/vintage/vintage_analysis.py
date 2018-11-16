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

# remove settingcopywithwarning
pd.options.mode.chained_assignment = None

# parameters for plotting
PLOT_PARAMS = {'test': {'s': 4, 'color': '#c04e01', 'alpha': 0.7},
               'train': {'s': 2, 'color': '#ffa756', 'alpha': 0.4},
               'actual line': {'lw': 2, 'color': '#0165fc',
                               'linestyle': '--'},
               'test roc curve': {'color': '#c04e01', 'lw': 2},
               'train roc curve': {'color': '#ffa756', 'lw': 2}}

# list of main variables
MAIN_VARS = ['ORIG_DTE', 'PRD', 'DFLT_AMT', 'NET_LOSS_AMT', 'dflt_pct',
             'net_loss_pct', 'did_dflt', 'dflt_loss_pct']

def get_fitted_model(train, test, choice):
    # print('\nFitting {0} with formula {1} and kwargs {2}'
    #       .format(choice[0], choice[1], choice[2]))
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


def fit_stage2(model_spec, train, test, pred_train=False,
               pred_input_data=None):
    model = get_fitted_model(train, test, model_spec)
    test_pred = model.make_pred()
    if pred_train:
        if pred_input_data is None:
            train_pred = model.make_pred(use_train=True)
        else:
            train_pred = model.make_pred(pred_input=pred_input_data)
        return np.exp(train_pred), np.exp(test_pred)
    else:
        return None, np.exp(test_pred)


def fit_stage3(model_spec, train, test, pred_train=False,
               pred_input_data=None):
    model = get_fitted_model(train, test, model_spec)
    if pred_train:
        if pred_input_data is None:
            return model.make_pred(use_train=True), model.make_pred()
        else:
            return (model.make_pred(pred_input=pred_input_data),
                    model.make_pred())
    else:
        return None, model.make_pred()


def fit_3stages(model_specs, train, test, pred_train=False):
    # stage 1
    train_PD_pred, test_PD_pred = fit_stage1(model_specs[0], train, test,
                                             pred_train)
    # stage 2
    train_pos = train[train['dflt_pct'] > 0]
    train_EAD_pred, test_EAD_pred = fit_stage2(model_specs[1], train_pos, test,
                                               pred_train, train)
    # stage 3
    train_LGD_pred, test_LGD_pred = fit_stage3(model_specs[2], train_pos, test,
                                               pred_train, train)
    if pred_train:
        return (train_PD_pred * train_EAD_pred * train_LGD_pred,
                test_PD_pred * test_EAD_pred * test_LGD_pred)
    else:
        return (None, test_PD_pred * test_EAD_pred * test_LGD_pred)


def main(args):
    ############################################################
    # READ/PROCESS/CLEAN DATA
    ############################################################
    # BASE_PATH = os.path.join(os.getcwd(), 'data')
    BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '../../data')

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
    df['did_dflt'] = 1*(df['dflt_pct'] > 0)
    df['net_loss_pct'] = 0
    df.loc[df['did_dflt'] == 1,
           'net_loss_pct'] = df['NET_LOSS_AMT'] / df['DFLT_AMT']

    # what we ultiamtely want to predict
    df['dflt_loss_pct'] = df['NET_LOSS_AMT'] / df['ORIG_AMT_sum']

    # create other variables
    regex = re.compile('_wv$')
    wghted_cols = [regex.sub('', c) for c in df.columns if regex.search(c)]
    for col in wghted_cols:
        df['{0}_cv'.format(col)] = np.sqrt(
            df['{0}_wv'.format(col)])/df['{0}_wm'.format(col)]

    # drop columns with too many NA values
    df = utility.drop_NA_cols(df)

    # remove all NA rows
    a = df.shape
    df.dropna(axis=0, how='any', inplace=True)
    print('Reduced rows from {0} -> {1}'.format(a, df.shape))

    # isolate covariates
    regex = re.compile('^MI_TYPE|^MI_PCT')
    all_vars = [
        v for v in df.columns if v not in MAIN_VARS and not regex.search(v)]

    ############################################################
    # MODEL SPECS
    ############################################################
    # Stage 1
    selected = ['UNEMP', 'HPI', 'rGDP', 'ORIG_AMT_sum', 'ORIG_CHN_R_wv',
                'LOAN_ID_count', 'MR', 'LIBOR', 'DTI_wm', 'AGE', 'CPI',
                'ORIG_RT_wm', 'ORIG_RT_cv', 'ORIG_CHN_R_wm', 'ORIG_RT_wv',
                'ORIG_CHN_R_cv', 'DTI_wv', 'OCLTV_wv',
                'ORIG_CHN_C_wv', 'ORIG_CHN_C_cv',
                'CSCORE_MN_wv', 'PURPOSE_R_cv', 'CSCORE_MN_wm', 'PURPOSE_R_wv']
    gbm_formula1 = 'did_dflt ~ -1 + {0}'.format(' + '.join(selected))
    kw1 = {'learning_rate': 0.1, 'max_depth': 3, 'max_features': 0.5,
           'min_impurity_decrease': 0.0001, 'n_estimators': 10, 'subsample': 0.1}

    # Stage 2
    # selected2 = ['np.log(ORIG_AMT_sum)*cr(AGE, df=5)', 'PROP_TYP_MH_wm*DTI_wm',
    #              'LIBOR', 'ORIG_TRM_cv', 'ORIG_RT_wm*MR',
    #              'ORIG_CHN_C_wm*np.log(ORIG_AMT_sum)', 'DTI_wm*UNEMP',
    #              '(OCC_STAT_P_wm+OCC_STAT_S_wm)', '(PURPOSE_P_wm+PURPOSE_R_wm)',
    #              '(ORIG_CHN_C_wm+ORIG_CHN_R_wm)',
    #              '(PROP_TYP_CP_wm+PROP_TYP_MH_wm+PROP_TYP_PU_wm+PROP_TYP_SF_wm)']
    # gam_formula2 = ('np.log(dflt_pct) ~ {0}'.format(' + '.join(selected2)))
    rfr_formula2 = ('np.log(dflt_pct) ~ -1 + {0}'
                    .format(' + '.join(all_vars)))
    kw2 = {'bootstrap': False, 'max_features': 0.333,
           'min_impurity_decrease': 1e-10, 'n_estimators': 20}

    # Stage 3
    # selected3 = ['AGE*(UNEMP + MR)', '(DTI_wm + DTI_wv) * UNEMP', 'HPI', 'LIBOR',
    #              'PROP_TYP_CP_wm + PROP_TYP_MH_wm', 'np.sqrt(LOAN_ID_count)']
    # gam_formula3 = ('net_loss_pct ~ {0}'.format(' + '.join(selected3)))
    rfr_formula3 = ('net_loss_pct ~ -1 + {0}'
                    .format(' + '.join(all_vars)))
    kw3 = {'n_estimators': 50, 'max_features': 0.1,
           'min_impurity_decrease': 1e-10}

    # define model_choice
    model_specs = [['GBC', gbm_formula1, kw1],
                   ['RFR', rfr_formula2, kw2],
                   ['RFR', rfr_formula3, kw3]]

    # create new folder if not exist
    if not os.path.exists(EXPORT_PATH):
        print('\nCreating results directory...')
        os.makedirs(EXPORT_PATH)
    output_dir_name = '.'.join([m[0] for m in model_specs] +
                               [FILENAME] +
                               [datetime.datetime.now()
                                .strftime("%y%m%d%H%M%S")])
    output_dir_path = os.path.join(EXPORT_PATH, output_dir_name)
    if not os.path.exists(output_dir_path):
        print('\nCreating directory at export location...')
        os.makedirs(output_dir_path)

    for train, test, trial_i in utility.train_test_splitter(df,
                                                            0.1, 'ORIG_DTE'):
        train_pos = train[train['dflt_pct'] > 0]

        # stage 1
        train['PD_pred'], test['PD_pred'] = fit_stage1(model_specs[0],
                                                       train, test, True)
        # stage 2
        train['EAD_pred'], test['EAD_pred'] = fit_stage2(model_specs[1],
                                                         train_pos, test,
                                                         True, train)
        # stage 3
        train['LGD_pred'], test['LGD_pred'] = fit_stage3(model_specs[2],
                                                         train_pos, test,
                                                         True, train)
        train['L_pred'] = train['PD_pred'] * \
            train['EAD_pred'] * train['LGD_pred']
        test['L_pred'] = test['PD_pred'] * test['EAD_pred'] * test['LGD_pred']
        test_pos = test[test['dflt_pct'] > 0]

        # get bootstraps
        print('\nRunning bootstraps...')
        btstrp_size = 4
        btstrp_3stage = bootstrap.get_btstrp(fit_3stages, model_specs,
                                             train, test, btstrp_size)
        # stage 1 bootstraps
        btstrp_stage1 = bootstrap.get_btstrp(fit_stage1, model_specs[0],
                                             train, test, 1)
        # stage 2 bootstraps
        btstrp_stage2 = bootstrap.get_btstrp(fit_stage2, model_specs[1],
                                             train_pos, test_pos, 1)
        # stage 2 bootstraps
        btstrp_stage3 = bootstrap.get_btstrp(fit_stage3, model_specs[2],
                                             train_pos, test_pos, 1)

        vin_id, x, lo_name, hi_name = 'ORIG_DTE', 'AGE', '2.5%', '97.5%'

        # compile bootstrapped data for each stages
        df_all = pd.concat([test[[vin_id, x, 'dflt_loss_pct', 'L_pred']],
                            btstrp_3stage], axis=1)
        df_stage1 = pd.concat([test[[vin_id, x, 'did_dflt', 'PD_pred']],
                               btstrp_stage1], axis=1)
        df_stage2 = pd.concat([test_pos[[vin_id, x, 'dflt_pct', 'EAD_pred']],
                               btstrp_stage2], axis=1)
        df_stage3 = pd.concat([test_pos[[vin_id, x,
                                         'net_loss_pct', 'LGD_pred']],
                               btstrp_stage3], axis=1)

        for date, dall in df_all.groupby(vin_id):
            print('\nGenerating plots for {0}...'.format(date))
            ds = []
            for df_ in [df_stage1, df_stage2, df_stage3]:
                to_append = df_[df_[vin_id] == date]
                # remove vin_id column
                del to_append[vin_id]
                ds.append(to_append)
            del dall[vin_id]

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # stage 1 roc
            ax, d1 = axes[0, 0], ds[0]
            fpr, tpr = utility.get_roc_curve(train['did_dflt'], train['PD_pred'])
            roc_auc = utility.get_auc(fpr, tpr)
            ax.plot(fpr, tpr, label='Train AUC = {0:.2f}'.format(roc_auc),
                    **PLOT_PARAMS['train roc curve'])

            fpr, tpr = utility.get_roc_curve(d1['did_dflt'], d1['PD_pred'])
            roc_auc = utility.get_auc(fpr, tpr)
            ax.plot(fpr, tpr, label='Test AUC = {0:.2f}'.format(roc_auc),
                    **PLOT_PARAMS['test roc curve'])

            ax.plot([-0.5, 1.5], [-0.5, 1.5], **PLOT_PARAMS['actual line'])
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Probability of Default: ROC curve')
            ax.legend(loc="lower right")

            # stage 2 predicted vs. actual
            ax, d2 = axes[0, 1], ds[1]

            ax.plot([-0.5, 1.5], [-0.5, 1.5], **PLOT_PARAMS['actual line'])
            c, d = train['EAD_pred'], train['dflt_pct']
            ax.scatter(c, d,
                       label='Train RMSE: {0:.2E}'.format(((c-d)**2).mean()),
                       **PLOT_PARAMS['train'])
            a, b = d2['EAD_pred'], d2['dflt_pct']
            ax.scatter(a, b,
                       label='Test RMSE: {0:.2E}'.format(((a-b)**2).mean()),
                       **PLOT_PARAMS['test'])
            xymax = min(0.5, max(a.max(), b.max()) * 1.05)
            xymin = max(-0.01, min(a.min(), b.min()) * 0.95)
            ax.set_xlim([xymin, xymax])
            ax.set_ylim([xymin, xymax])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Exposure at Default: % balance at default')
            ax.legend(loc="lower right")

            # # stage 2 bootstrap actual vs. predicted
            # ax, d2 = axes[0, 1], ds[1]
            # bootstrap.plot_btstrp(d2, 'dflt_pct', 'dflt_pct', 'EAD_pred',
            #                       lo_name, hi_name, ax=ax)
            # ax.set_xlabel('Actual')
            # ax.set_ylabel('Predicted')
            # ax.set_title('Exposure at Default: % balance at default')
            # ax.legend(loc="lower right")

            # stage 3 predicted vs. actual
            ax, d3 = axes[1, 0], ds[2]

            ax.plot([-10, 10], [-10, 10], **PLOT_PARAMS['actual line'])
            c, d = train['LGD_pred'], train['net_loss_pct']
            ax.scatter(c, d,
                       label='Train RMSE: {0:.2E}'.format(((c-d)**2).mean()),
                       **PLOT_PARAMS['train'])
            a, b = d3['LGD_pred'], d3['net_loss_pct']
            ax.scatter(a, b,
                       label='Test RMSE: {0:.2E}'.format(((a-b)**2).mean()),
                       **PLOT_PARAMS['test'])

            xymax = min(2, max(a.max(), b.max()) * 1.05)
            xymin = max(-0.5, min(a.min(), b.min()) * 0.95)
            ax.set_xlim([xymin, xymax])
            ax.set_ylim([xymin, xymax])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Loss given Default: % lost at default')
            ax.legend(loc="lower right")

            # # stage 3 bootstrap actual vs. predicted
            # ax, d3 = axes[1, 0], ds[2]
            # bootstrap.plot_btstrp(d3, 'net_loss_pct', 'net_loss_pct', 'LGD_pred',
            #                       lo_name, hi_name, ax=ax)
            # ax.set_xlabel('Actual')
            # ax.set_ylabel('Predicted')
            # ax.set_title('Loss given Default: % lost at default')
            # ax.legend(loc="lower right")

            # 3 stage bootstrap cumulative
            ax = axes[1, 1]
            cumsum = dall.set_index(x).cumsum()
            bootstrap.plot_btstrp(cumsum.reset_index(), x, 'dflt_loss_pct',
                                  'L_pred', lo_name, hi_name, ax=ax)
            ax.set_xlabel('Age of loan')
            ax.set_ylabel('% of ORIG_AMT lost')
            loan_count = test.loc[test.ORIG_DTE ==
                                  date, 'LOAN_ID_count'].values[0]
            ax.set_title(
                'Projected Loss: {0} loans from {1}'.format(loan_count, date))
            ax.legend(loc="lower right")
            # plot settings
            plt.tight_layout()
            plotname = '{2}.{1}.{0}.png'.format(loan_count, date, trial_i)
            plt.savefig(os.path.join(output_dir_path, plotname),
                        dpi=300, bbox_inches='tight')
            # plt.show()
            plt.clf()

        # # generate plots with all vintages grouped
        # ds = [df_stage1, df_stage2, df_stage3]
        # dall = df_all
        # del dall[vin_id]
        # fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # # stage 1 roc
        # ax, d1 = axes[0, 0], ds[0]
        # fpr, tpr = utility.get_roc_curve(d1['did_dflt'], d1['PD_pred'])
        # roc_auc = utility.get_auc(fpr, tpr)
        # ax.plot([-0.5, 1.5], [-0.5, 1.5], **PLOT_PARAMS['actual line'])
        # ax.plot(fpr, tpr, label='AUC = {0:.2f}'.format(roc_auc),
        #         **PLOT_PARAMS['roc curve'])
        # ax.set_xlim([-0.05, 1.05])
        # ax.set_ylim([-0.05, 1.05])
        # ax.set_xlabel('False Positive Rate')
        # ax.set_ylabel('True Positive Rate')
        # ax.set_title('Probability of Default: ROC curve')
        # ax.legend(loc="lower right")

        # # stage 2 predicted vs. actual
        # ax, d2 = axes[0, 1], ds[1]

        # ax.plot([-0.5, 1.5], [-0.5, 1.5], **PLOT_PARAMS['actual line'])
        # c, d = train['EAD_pred'], train['dflt_pct']
        # ax.scatter(c, d,
        #            label='Train RMSE: {0:.2E}'.format(((c-d)**2).mean()),
        #            **PLOT_PARAMS['train'])
        # a, b = d2['EAD_pred'], d2['dflt_pct']
        # ax.scatter(a, b,
        #            label='Test RMSE: {0:.2E}'.format(((a-b)**2).mean()),
        #            **PLOT_PARAMS['test'])
        # xymax = max(a.max(), b.max()) * 1.05
        # xymin = min(a.min(), b.min()) * 0.95
        # ax.set_xlim([xymin, xymax])
        # ax.set_ylim([xymin, xymax])
        # ax.set_xlabel('Predicted')
        # ax.set_ylabel('Actual')
        # ax.set_title('Exposure at Default: % balance at default')
        # ax.legend(loc="lower right")

        # # stage 3 predicted vs. actual
        # ax, d3 = axes[1, 0], ds[2]

        # ax.plot([-10, 10], [-10, 10], **PLOT_PARAMS['actual line'])
        # c, d = train['LGD_pred'], train['net_loss_pct']
        # ax.scatter(c, d,
        #            label='Train RMSE: {0:.2E}'.format(((c-d)**2).mean()),
        #            **PLOT_PARAMS['train'])
        # a, b = d3['LGD_pred'], d3['net_loss_pct']
        # ax.scatter(a, b,
        #            label='Test RMSE: {0:.2E}'.format(((a-b)**2).mean()),
        #            **PLOT_PARAMS['test'])

        # xymax = max(a.max(), b.max()) * 1.05
        # xymin = min(a.min(), b.min()) * 0.95
        # ax.set_xlim([xymin, xymax])
        # ax.set_ylim([xymin, xymax])
        # ax.set_xlabel('Predicted')
        # ax.set_ylabel('Actual')
        # ax.set_title('Loss given Default: % lost at default')
        # ax.legend(loc="lower right")

        # # 3 stage bootstrap cumulative
        # ax = axes[1, 1]
        # cumsum = dall.set_index(x).cumsum()
        # bootstrap.plot_btstrp(cumsum.reset_index(), x, 'dflt_loss_pct',
        #                       'L_pred', lo_name, hi_name, ax=ax)
        # ax.set_xlabel('Age of loan')
        # ax.set_ylabel('% of ORIG_AMT lost')
        # uniq_dtes = test.ORIG_DTE.unique()
        # str_uniq_dtes = ', '.join([str(d) for d in uniq_dtes])
        # loan_count = sum([test.loc[test.ORIG_DTE == date,
        #                            'LOAN_ID_count'].values[0]
        #                   for date in uniq_dtes])
        # ax.set_title('Projected Loss: {0} loans from {1}...'
        #              .format(loan_count, str_uniq_dtes[:17]))
        # ax.legend(loc="lower right")
        # # plot settings
        # plt.tight_layout()
        # plotname = '{2}.{1}.{0}.png'.format(loan_count, str_uniq_dtes,
        #                                     trial_i)
        # plt.savefig(os.path.join(main_dir_path, plotname),
        #             dpi=300, bbox_inches='tight')
        # # plt.show()
        # plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probability of default model for vintage data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataname', type=str, nargs='?', default='fannie_mae_data',
                        help='name of data folder')
    parser.add_argument('filename', type=str, help='name of data file')
    args = parser.parse_args()
    main(args)
