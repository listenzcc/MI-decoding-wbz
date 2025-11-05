"""
File: decode-fbcsp.py
Author: Chuncheng Zhang
Date: 2025-11-05
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read data and FBCSP it.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-05 ------------------------
# Requirements and constants
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from mne.decoding import CSP, Scaler

from util.easy_import import *
from collect_data import find_bdf_files, read_eeg_data, MyData

# %%
DATA_DIR = Path('./raw/MI-data-QS')
SUBJECT = 'S1'

if len(sys.argv) > 2 and sys.argv[1] == '-s':
    SUBJECT = sys.argv[2]

# %%
FREQ_RANGES = [(e, e+4) for e in range(1, 45, 2)]

# %%
OUTPUT_DIR = Path(f'./data/results/fbcsp/{SUBJECT}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2025-11-05 ------------------------
# Function and class


def decode_on_band(l_freq, h_freq, epochs, groups, labels):
    # Setup
    baseline = (-1, 0)
    tmin, tmax = 0, 4

    # Work on the copied epochs
    epochs = epochs.copy()
    epochs.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
    epochs.apply_baseline(baseline)
    epochs.crop(tmin=tmin, tmax=tmax)

    # Prepare for decoding
    X = epochs.get_data()
    y = labels

    cv = LeaveOneGroupOut()

    clf = make_pipeline(
        Scaler(epochs.info),
        CSP(),
        LogisticRegression()
    )

    y_proba = cross_val_predict(
        estimator=clf, X=X, y=y, groups=groups, cv=cv, method='predict_proba')

    y_pred = np.argmax(y_proba, axis=1) + 1

    return {
        'l_freq': l_freq,
        'h_freq': h_freq,
        'y_proba': y_proba,
        'y_pred': y_pred,
    }


# %% ---- 2025-11-05 ------------------------
# Play ground
table = find_bdf_files(DATA_DIR).query(f'subject == "{SUBJECT}"')
print(table)

mds = []
for i, se in tqdm(table.iterrows(), 'Load data'):
    mds.append(MyData(read_eeg_data(se), se))
print(mds)

epochs = mne.concatenate_epochs([e.epochs for e in mds])
groups = np.concat([np.zeros((len(e.epochs), )) +
                   i for i, e in enumerate(mds)]).ravel()
labels = epochs.events[:, -1]
print(f'{epochs=}, {groups=}, {labels=}')

result = {
    'subject': SUBJECT,
    'y_true': labels,
    'bands': []
}
for l_freq, h_freq in tqdm(FREQ_RANGES, 'Decoding on bands'):
    res = decode_on_band(l_freq, h_freq, epochs, groups, labels)
    result['bands'].append(res)


joblib.dump(result, OUTPUT_DIR.joinpath('fbcsp-decoding-results.dump'))

logger.info(f'Done with {__file__}')


# %% ---- 2025-11-05 ------------------------
# Pending

# %% ---- 2025-11-05 ------------------------
# Pending
