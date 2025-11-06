"""
File: summary-fbcnet.py
Author: Chuncheng Zhang
Date: 2025-11-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the FBCNet results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-06 ------------------------
# Requirements and constants
from sklearn.metrics import classification_report

from util.easy_import import *

# %%
DATA_DIR = Path('./data/results/fbcnet')
OUTPUT_DIR = DATA_DIR

# %% ---- 2025-11-06 ------------------------
# Function and class


def parse_result(result: dict):
    y_true = result['y_true']
    y_pred = result['y_pred']
    subject = result['subject']

    y_filter = [not t == 3 and not p == 3 for t, p in zip(y_true, y_pred)]
    y_true = y_true[y_filter]
    y_pred = y_pred[y_filter]

    rep = classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True)

    rep.update({'subject': subject, 'samples': len(y_true)})
    return rep


# %% ---- 2025-11-06 ------------------------
# Play ground
dump_files = sorted(DATA_DIR.rglob('fbcnet-decoding-results.dump'))
reports = []

for d_file in tqdm(dump_files, 'Dealing with files'):
    result = joblib.load(d_file)
    report = parse_result(result)
    print(report)
    reports.append(report)

df = pd.DataFrame(reports)
print(df)
print(df.mean(numeric_only=True))
print(sorted(df['accuracy']))

# %% ---- 2025-11-06 ------------------------
# Pending


# %% ---- 2025-11-06 ------------------------
# Pending
