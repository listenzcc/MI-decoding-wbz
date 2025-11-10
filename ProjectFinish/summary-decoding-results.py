"""
File: summary-decoding-results.py
Author: Chuncheng Zhang
Date: 2025-11-07
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the FBCSP results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-07 ------------------------
# Requirements and constants
from util.easy_import import *

FBCSP_NAME = 'fbcsp-info'
METHOD = 'default'

FBCSP_NAME = 'fbcsp-vote'
METHOD = 'softvote'

FBCSP_NAME = 'fbcsp-vote'
METHOD = 'hardvote'

print(sys.argv)

if sys.argv[1] == '-n':
    FBCSP_NAME = sys.argv[2]

if sys.argv[3] == '-m':
    METHOD = sys.argv[4]

DATA_DIR = Path(f'./data/exp_record/results/{FBCSP_NAME}')

# %% ---- 2025-11-07 ------------------------
# Function and class


# %% ---- 2025-11-07 ------------------------
# Play ground
data_table = pd.DataFrame([(e.parent.name[:-1], e.parent.name[-1], e.stem.split('_')[-1], e)
                           for e in DATA_DIR.rglob('*.dump')],
                          columns=['subject', 'day', 'run', 'path'])

print('='*80)
print(f'{FBCSP_NAME=}, {METHOD=}')
print('Example obj:')
print(joblib.load(data_table.iloc[0]['path']))

if any([
    FBCSP_NAME == 'fbcsp-info' and METHOD in ['default'],
    FBCSP_NAME == 'fbcsp-info-cvruns' and METHOD in ['default'],
]):
    data_table['acc'] = data_table['path'].map(
        lambda f: np.mean(joblib.load(f)))

elif any([
    FBCSP_NAME == 'fbcsp-vote' and METHOD in ['softvote', 'hardvote'],
    FBCSP_NAME == 'fbcnet' and METHOD in ['fbcnet'],
    FBCSP_NAME == 'fbcnet-cvruns' and METHOD in ['fbcnet'],
]):
    data_table['acc'] = data_table['path'].map(lambda f: np.mean(
        [e[1] for e in joblib.load(f) if e[0] == METHOD]))

else:
    raise ValueError(f'Known param: {FBCSP_NAME=}, {METHOD=}')

data_table['method'] = METHOD

data_table = data_table.sort_values(by='acc', ascending=False)

for c in ['day', 'run']:
    data_table[c] = data_table[c].map(int)

print(f'{len(data_table)=}')

# Group by subject and day
group = data_table.groupby(['subject', 'day', 'method'])
# print(group.mean('acc'))

# Mean acc by day
group = data_table.groupby(['day', 'method'])
print(group.mean('acc'))

# Find subject's top 10 runs
group = data_table.groupby(['subject', 'method'])
top10 = group.head(10)
# print(top10)
print(top10.mean(numeric_only=True))

# %% ---- 2025-11-07 ------------------------
# Pending


# %% ---- 2025-11-07 ------------------------
# Pending
