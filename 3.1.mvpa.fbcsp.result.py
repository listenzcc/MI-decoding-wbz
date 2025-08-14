"""
File: 3.1.mvpa.fbcsp.result.py
Author: Chuncheng Zhang
Date: 2025-08-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the FBCSP voting results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-08-14 ------------------------
# Requirements and constants
from sklearn import metrics
from util.easy_import import *
from util.io.file import load

# Setup
data_directory = Path('./data/fbcsp')

compile = re.compile(
    r'^(?P<sn>[A-Za-z0-9]+)-freq-CSP-results.dump')

# %% ---- 2025-08-14 ------------------------
# Function and class


def find_csp_voting_files(src: Path):
    ps = list(data_directory.rglob('*-freq-CSP-results.dump'))
    parsed = [compile.search(p.name).groupdict() for p in ps]
    return [(info, p) for info, p in zip(parsed, ps)]


# %% ---- 2025-08-14 ------------------------
# Read files and summary voting results
data = []
for info, p in find_csp_voting_files(data_directory):
    print(info)
    print(p)

    d = load(p)
    y_true = d['y_true']

    # y_preds shape: bands x samples
    y_preds = [v['y_pred'] for k, v in d.items() if isinstance(k, int)]

    # y_probas shape: bands x samples x classes
    y_probas = [v['y_proba'] for k, v in d.items() if isinstance(k, int)]

    # Map y: 0 -> 10, 1 -> 11
    y_pred = np.argmax(np.prod(np.array(y_probas), axis=0), axis=1) + 10

    for i, y_pred in enumerate(y_preds):
        acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        data.append({'subject': info['sn'], 'acc': acc, 'freqIdx': i})

    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    data.append({'subject': info['sn'], 'acc': acc, 'freqIdx': 'vote'})

data = pd.DataFrame(data)
print(data)

group = data.groupby(by=['freqIdx', 'subject'])
print(group['acc'].mean())

print(data.query('freqIdx=="vote"'))
print('Mean acc=', data.query('freqIdx=="vote"')['acc'].mean())

# %% ---- 2025-08-14 ------------------------
# Pending


# %% ---- 2025-08-14 ------------------------
# Pending
