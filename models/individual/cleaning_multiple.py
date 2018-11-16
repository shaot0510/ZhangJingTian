import os
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width', 100)

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


# PATH1 = '~/Google Drive/PWBM/CECL/data'
PATH1 = 'data'
filesize = '500K'
incr = 3
quarters = [4]

# YEARS = ['{0}Q{1}'.format(y, q)
#          for y in range(2000, 2017)
#          for q in range(1, 5)] + ['2017Q1']
YEARS = ['{0}Q{1}'.format(y, q) for y in range(2000, 2017, incr) for q in quarters]
date_vars = ['ORIG_DTE', 'FRST_DTE', 'Monthly.Rpt.Prd', 'Maturity.Date',
             'LPI_DTE', 'FCC_DTE', 'DISP_DT', 'LAST_DTE']
df_indvs = pd.DataFrame()
for YEAR in YEARS:
    print('\nReading {0}'.format(YEAR))
    # read monthly loan data
    PATH2 = os.path.join(PATH1, 'fannie_mae/raw/all')
    filename = '.'.join([filesize, YEAR, 'csv'])
    df = pd.read_csv(os.path.join(PATH2, filename))

    # delete unnecessary
    del df['Loan.Age']

    # change date formats
    for date_var in date_vars:
        df[date_var] = lookup(df[date_var])

    # rename columns
    df.rename(index=str, columns={'Monthly.Rpt.Prd':'PRD'}, inplace=True)

    temp = df.groupby('LOAN_ID').last()

    CreditEvents = ["F", "S", "T", "N"]
    print('\nCreating did_dflt')
    temp['did_dflt'] = 0
    temp.loc[temp['LAST_STAT'].isin(CreditEvents), 'did_dflt'] = 1
    temp['DFLT_AMT'] = temp['Fin_UPB'] * temp['did_dflt']
    temp['NET_LOSS_AMT'] = temp['NET_LOSS'] * temp['did_dflt']
    df_indvs = pd.concat([df_indvs, temp], axis=0)

# EXPORT
PATH3 = os.path.join(PATH1, 'fannie_mae', 'clean')
export_name= 'indv_{0}.{1}.csv'.format(filesize, '.'.join([YEARS[0], YEARS[-1], 'inc{0}'.format(incr)]))
df_indvs.to_csv(os.path.join(PATH3, export_name))
