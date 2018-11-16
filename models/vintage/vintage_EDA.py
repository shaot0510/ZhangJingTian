import os
import argparse
import importlib
import math
import time
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import models, bootstrap, utility


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

############################################################
# READ/PROCESS/CLEAN DATA
############################################################
parser = argparse.ArgumentParser(description='Probability of default model for vintage data.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataname', type=str, nargs='?', default='fannie_mae_data',
                    help='name of data folder')
parser.add_argument('filename', type=str, help='name of data file')
args = parser.parse_args(['vint_filelist_500.csv'])

BASE_PATH = os.path.join(os.getcwd(), 'data')
# BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
#                     '../../data')

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

################################################################################
############################## EXPLORING ANALYSIS ##############################
################################################################################

############################## LOSS ACROSS YEARS ##############################
df['ORIG_YR'] = df.ORIG_DTE.apply(lambda x: x.year)

sns.catplot(x="ORIG_YR", y="net_loss_pct", kind="boxen", data=df)

sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="ORIG_YR", hue="ORIG_YR", aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "net_loss_pct", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "net_loss_pct", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "net_loss_pct")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
for ax in g.axes:
    ax[0].set_xlim([-0.8, 1.5])
plt.show()

sns.set(style="white")
yr_dict = {2000: '00~04',
           2001: '00~04',
           2002: '00~04',
           2003: '00~04',
           2004: '00~04',
           2005: '05~08',
           2006: '05~08',
           2007: '05~08',
           2008: '05~08',
           2009: '09~12',
           2010: '09~12',
           2011: '09~12',
           2012: '09~12'}

df['ORIG_GRP'] = df.ORIG_YR.apply(lambda x: yr_dict[x])
for yr_grp, df_in_yr in df.groupby('ORIG_GRP'):
    sns.distplot(df_in_yr['net_loss_pct'], label=yr_grp)
plt.legend()

for yr_grp, df_in_yr in df.groupby('ORIG_GRP'):
    no_zeros = df_in_yr.loc[df_in_yr.net_loss_pct != 0, 'net_loss_pct']
    sns.distplot(no_zeros, label=yr_grp)
plt.legend()

############################## AGE ##############################
dflt_df = df[df.did_dflt == 1]
sns.scatterplot(x="AGE", y="dflt_pct",
                marker='+',
                data=dflt_df)

sns.scatterplot(x="AGE", y="dflt_pct",
                hue="ORIG_GRP",
                palette="ch:r=-.2,d=.3_r",
                linewidth=0,
                s=10,
                alpha=0.9,
                data=dflt_df)

############################## MACROS ##############################
sns.scatterplot(x="UNEMP", y="dflt_pct",
                hue="ORIG_GRP",
                palette="ch:r=-.2,d=.3_r",
                linewidth=0,
                s=10,
                alpha=0.9,
                data=dflt_df)

crisis_df = dflt_df[dflt_df.ORIG_GRP == '05~08']
sns.jointplot(x='UNEMP', y='dflt_pct', data=crisis_df, kind="reg")

############################## QUESTION ##############################
# Define a month as IS_REC = True if UNEMP increased by more than 1.5% compared to 12 months prior
# Plot IS_REC vs. net_loss_pct
