"""
File: summary-fbcsp.py
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

DATA_DIR = Path('./data/exp_record/results/fbcsp')

# %% ---- 2025-11-07 ------------------------
# Function and class


# %% ---- 2025-11-07 ------------------------
# Play ground
data_table = pd.DataFrame([(e.parent.name[:-1], e.parent.name[-1], e.stem.split('_')[-1], e)
                           for e in DATA_DIR.rglob('*.dump')],
                          columns=['subject', 'day', 'run', 'path'])

data_table['acc'] = data_table['path'].map(lambda f: np.mean(joblib.load(f)))
data_table = data_table.sort_values(by='acc', ascending=False)

for c in ['day', 'run']:
    data_table[c] = data_table[c].map(int)

# Group by subject and day
group = data_table.groupby(['subject', 'day'])
print(group.mean('acc'))

# Mean acc by day
group = data_table.groupby('day')
print(group.mean('acc'))

# Find subject's top 10 runs
group = data_table.groupby('subject')
top10 = group.head(10)
print(top10)
print(top10.mean(numeric_only=True))

# %% ---- 2025-11-07 ------------------------
# Pending


# %% ---- 2025-11-07 ------------------------
# Pending
