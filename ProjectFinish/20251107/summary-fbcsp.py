"""
File: summary-fbcsp.py
Author: Chuncheng Zhang
Date: 2025-11-05
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the results of FBCSP decoding.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-05 ------------------------
# Requirements and constants
from sklearn.metrics import classification_report, confusion_matrix

from util.easy_import import *

# %%
DATA_DIR = Path('./data/results/fbcsp')
OUTPUT_DIR = DATA_DIR

# %% ---- 2025-11-05 ------------------------
# Function and class


def parse_result(result: dict):
    # y_true shape is (n_samples, )
    y_true = result['y_true']
    # arr_proba shape is (n_bands, n_samples, n_labels=3)
    arr_proba = np.array([e['y_proba'] for e in result['bands']])
    # arr_pred shape is (n_bands, n_samples)
    arr_pred = np.array([e['y_pred'] for e in result['bands']])
    print(f'{y_true.shape=}, {arr_proba.shape=}, {arr_pred.shape=}')

    report = []
    mats = []

    def filter_y(y_true, y_pred):
        y_filter = [t in [1, 2] and p in [1, 2]
                    for t, p in zip(y_true, y_pred)]
        y_true = y_true[y_filter]
        y_pred = y_pred[y_filter]
        return y_true, y_pred

    # Hard vote
    counts = np.array([np.sum(arr_pred == e, axis=0) for e in [1, 2, 3, 4, 5]])
    y_pred = np.argmax(counts, axis=0) + 1
    y_true = result['y_true'].copy()

    y_true, y_pred = filter_y(y_true, y_pred)

    rep = classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True)
    rep.update({'subject': result['subject'],
               'samples': len(y_true),
                'method': 'hardVote'})
    report.append(rep)
    # mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # mats.append(mat)

    # Soft vote
    probs = np.prod(arr_proba, axis=0)
    y_pred = np.argmax(probs, axis=1) + 1
    y_true = result['y_true'].copy()

    y_true, y_pred = filter_y(y_true, y_pred)

    rep = classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True)
    rep.update({'subject': result['subject'],
               'samples': len(y_true),
                'method': 'softVote'})
    report.append(rep)
    mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    mats.append(mat)

    return report, np.array(mats)


# %% ---- 2025-11-05 ------------------------
# Play ground
dump_files = sorted(DATA_DIR.rglob('fbcsp-decoding-results.dump'))

reports = []
mats = []
for d_file in tqdm(dump_files, 'Dealing with files'):
    result = joblib.load(d_file)
    report, mat = parse_result(result)
    print(report)
    reports.extend(report)
    mats.extend(mat)
mats = np.array(mats)

df = pd.DataFrame(reports)
print(df)
print(df.groupby('method').mean(numeric_only=True))
# print(mats.squeeze())

# %% ---- 2025-11-05 ------------------------
# Pending


# %% ---- 2025-11-05 ------------------------
# Pending
