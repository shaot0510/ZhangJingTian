import sys
import argparse
import os

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    valid = s[~s.isna()]
    dates = {date: pd.to_datetime(date, infer_datetime_format=True).to_period('m')
             for date in valid.unique()}
    dates[np.nan] = np.nan
    return s.map(dates)


def main(args):
    lower_lim = args.lower_lim    
    PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '../../data', args.dataname)
    # PATH = os.path.join(os.getcwd(), 'data', 'fannie_mae_data')

    with open(os.path.join(PATH, args.filelist)) as f:
        filelist, yearlist = [], []
        for line in f:
            year_files = line.split()
            try:
                yearlist.append(int(year_files[0]))
            except ValueError as ex:
                print('ValueError: First value in each row of filelist must be a year')
                sys.exit(1)
            filelist.append(year_files[1:])

    with open(os.path.join(PATH, args.varlist)) as f:
        varlist = {'CAT':[], 'CONT':[]}
        for line in f:
            type_vars = line.split()
            typekey = type_vars[0].upper().replace(' ', '')
            try:
                varlist[typekey] = type_vars[1:]
            except KeyError:
                print('KeyError: Each row of varlist must start with the keyword CAT or CONT')
                sys.exit(1)

    cat_vars, cont_vars = varlist['CAT'], varlist['CONT']
    del varlist

    print('=' * 80)
    print('\nSUMMARY:')
    print('Data from {0} with filelist from {1}'.format(args.dataname, args.filelist))
    print('Continuous variables: {0}\nCategorical variables: {1}'.format(', '.join(cont_vars),
                                                                         ', '.join(cat_vars)))
    print('Lower limit set at {0} with {1} years from {2} to {3}'.format(lower_lim, len(yearlist),
                                                                         yearlist[0], yearlist[-1]))
    ############################################################
    # FIX: single vintage has multiple AGE values (2014~2015)
    # FIX: compute mean, var in addition to wm, wv
    # Compute estimated balance at each month
    ############################################################

    PATH1 = os.path.join(PATH, 'clean')

    df_vintages = pd.DataFrame()
    for i in range(len(filelist)):
        filebatch = filelist[i]
        print('\n' + '=' * 80)
        print('=' * 80)
        print('\nProcessing current batch...: {0}'.format(filebatch))
        df = pd.DataFrame()
        for filename in filebatch:
            print('\nReading {0}...'.format(filename))
            # read monthly loan data
            to_concat = pd.read_csv(os.path.join(PATH1, filename),
                                    engine='c')

            df = pd.concat([df, to_concat], axis=0)
        print('Total number of rows: {0}'.format(df.shape[0]))

        df['ORIG_DTE'] = lookup(df['ORIG_DTE'])

        print('\nReducing/modifying df by year...')
        year = yearlist[i]
        # remove vintages not from the year in list    
        df['yORIG_DTE'] = df['ORIG_DTE'].apply(lambda x: int(x.year))
        a = df.shape
        df = df[df['yORIG_DTE'] == year]
        print('With year restriction, retained {0:.2f}% of {1} loans'
              .format(100 * df.shape[0]/a[0], a[0]))
        del df['yORIG_DTE']

        # REMOVE VINTAGES with < lower_lim loans
        df['strORIG_DTE'] = df['ORIG_DTE'].astype(str)
        num_loans = df.groupby('strORIG_DTE')['LOAN_ID'].nunique()

        vintages = num_loans.index[num_loans > lower_lim]

        if len(vintages) == 0:
            print('ValueError: No vintages selected. Set lower LOWER_LIM or provide bigger dataset')
            sys.exit(1)

        print('\nIncluded Vintages: {0}'.format(', '.join([v for v in vintages])))
        print(num_loans)

        df_red = df[df['strORIG_DTE'].isin(vintages)]
        del df_red['strORIG_DTE']

        # change date format
        df_red['PRD'] = lookup(df_red['PRD'])

        # create column of dflt/loss amount
        print('\nCreating ORIG_data...')

        df_red['strPRD'] = df_red['PRD'].astype(str)
        ORIG_vars = (['strPRD', 'ORIG_AMT', 'ORIG_DTE', 'DID_DFLT'] +
                     # cat, cont vars used later
                     cat_vars + cont_vars)
        ORIG_data = df_red.groupby('LOAN_ID')[ORIG_vars].last().reset_index()
        # delete DID_DFLT in original df
        del df_red['DID_DFLT']

        to_merge = ORIG_data[['LOAN_ID', 'strPRD', 'DID_DFLT']]

        df_red = df_red.merge(to_merge, how='left', on=['LOAN_ID', 'strPRD'])
        df_red['DID_DFLT'] = df_red['DID_DFLT'].fillna(value=0)
        df_red['DFLT_AMT'] = df_red['FIN_UPB'] * df_red['DID_DFLT']
        df_red['NET_LOSS_AMT'] = df_red['NET_LOSS'] * df_red['DID_DFLT']

        def wm_wv(x):
            nonna_inds = x[~x.isna()].index
            values = x[nonna_inds]
            wghts = ORIG_data.loc[nonna_inds, "ORIG_AMT"]
            wghtd_avg = np.average(values, weights=wghts)
            wghtd_var = np.average((values-wghtd_avg)**2, weights=wghts)
            return (wghtd_avg, wghtd_var)

        f = {
            'LOAN_ID': 'count',
            'ORIG_AMT': 'sum',
        }

        for col in cont_vars:
            f[col] = wm_wv

        for cat in cat_vars:
            dummies = pd.get_dummies(ORIG_data[cat], prefix=cat, drop_first=True)
            ORIG_data = pd.concat([ORIG_data, dummies], axis=1, join='outer')
            # del ORIG_data[cat]
            for col in dummies.columns:
                f[col] = wm_wv

        print('\nAggregating vintage data...')
        vintage_data_unsplit = ORIG_data.groupby('ORIG_DTE').agg(f)
        vintage_data = pd.DataFrame(index=vintage_data_unsplit.index)
        for col in vintage_data_unsplit.columns:
            if f[col] == wm_wv:
                split_cols = vintage_data_unsplit[col].apply(pd.Series)
                split_cols.columns = [col+'_wm', col+'_wv']
            else:
                split_cols = vintage_data_unsplit[col]
                split_cols.name = '{0}_{1}'.format(col, f[col])
            vintage_data = pd.concat([vintage_data, split_cols], axis=1)
        vintage_data.reset_index(inplace=True)

        df_vintage = df_red.groupby(['ORIG_DTE', 'PRD'])[['DFLT_AMT',
                                                         'NET_LOSS_AMT']].sum()
        df_vintage.reset_index(inplace=True)

        print('\nMerging vintage datasets...')
        df_vintage = df_vintage.merge(vintage_data, on='ORIG_DTE')
        df_vintages = pd.concat([df_vintages, df_vintage], axis=0)

    # EXPORT
    print('\n' + '=' * 80)
    print('=' * 80)
    print('\nExporting file to {0}...'.format(os.path.join('data',
                                                           args.dataname, 'vintage_analysis',
                                                           'data')))
    VINTAGE_PATH = os.path.join(PATH, 'vintage_analysis')
    EXPORT_PATH = os.path.join(VINTAGE_PATH, 'data')
    # create new folder if not exist
    if not os.path.exists(VINTAGE_PATH):
        print('Creating main directory for vintage analysis')
        os.makedirs(VINTAGE_PATH)
    if not os.path.exists(EXPORT_PATH):
        print('Creating directory at export location')
        os.makedirs(EXPORT_PATH)
        
    export_name = 'vintage_{0}_{1}.csv'.format(os.path.splitext(args.filelist)[0],
                                               lower_lim)
    df_vintages.to_csv(os.path.join(EXPORT_PATH, export_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate loan data by vintage.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataname', type=str, nargs='?', default='fannie_mae_data',
                        help='name of data folder')
    parser.add_argument('filelist', type=str, nargs='?', default='filelist.txt',
                        help='text file containing the years and file names to read')
    parser.add_argument('varlist', type=str, nargs='?', default='varlist.txt',
                        help='text file containing a list of categorical and continuous loan variables')    
    parser.add_argument('-l', '--lower_lim', type=int, default=500,
                        help='minimum number of loans per vintage')
    args = parser.parse_args()
    main(args)
