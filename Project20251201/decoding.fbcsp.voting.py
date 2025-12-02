"""
File: decoding.fbcsp.voting.py
Author: Chuncheng Zhang
Date: 2025-12-01
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding with FBCSP voting method.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-12-01 ------------------------
# Requirements and constants
from util.easy_import import *
from stratified_cv_groups import stratified_cv_groups

from read_data import load_epochs_data

from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from mne.decoding import CSP, Scaler


# %%
DATA_DIR = Path('./data/decoding-FBCSP-voting')
DATA_DIR.mkdir(exist_ok=True, parents=True)

RAW_DIR = Path('./raw/wbz-20251201-data')

FREQ_RANGES = [(e, e+4) for e in range(1, 45, 2)]

# %%
# Check results
if False:
    dump_files = DATA_DIR.rglob('*.joblib')
    for fname in dump_files:
        obj = joblib.load(fname)
        y_true = obj['y_true']
        y_probas = []
        for i, freqs in enumerate(obj['freqs']):
            y_probas.append(obj[i]['y_proba'])
        y_probas = np.array(y_probas)
        y_proba = np.prod(y_probas, axis=0)
        y_pred = np.argmax(y_proba, axis=1) + 1
        acc = np.mean(y_pred == y_true)
        print(acc)

# %% ---- 2025-12-01 ------------------------
# Function and class


def decoding_with_FBCSP_voting(epochs):
    y = epochs.events[:, -1]
    groups = stratified_cv_groups(y)
    # init scores
    freq_CSP_results = {
        'y_true': y,
        'freqs': FREQ_RANGES
    }

    # Loop through each frequency range of interest
    for freqIdx, (fmin, fmax) in tqdm(enumerate(FREQ_RANGES), total=len(FREQ_RANGES)):
        filter_kwargs = {'l_freq': fmin, 'h_freq': fmax, 'n_jobs': n_jobs}
        epochs_filter = epochs.copy()
        epochs_filter.filter(**filter_kwargs)
        epochs_filter.apply_baseline((-1, 0))
        epochs_filter.crop(tmin=0, tmax=4)

        X = epochs_filter.get_data(copy=False)

        cv = 5

        clf = make_pipeline(
            Scaler(epochs_filter.info),
            CSP(),
            LogisticRegression(),
        )
        y_proba = cross_val_predict(
            estimator=clf, X=X, y=y, groups=groups, cv=cv, method='predict_proba')

        y_pred = np.argmax(y_proba, axis=1) + 1

        # y_pred = cross_val_predict(estimator=clf, X=X, y=y, groups=groups, cv=cv)

        print(classification_report(y_true=y, y_pred=y_pred))
        freq_CSP_results[freqIdx] = {
            'fmin': fmin,
            'fmax': fmax,
            'y_proba': y_proba,
            'y_pred': y_pred}

    return freq_CSP_results

# %%
# Check results


# %% ---- 2025-12-01 ------------------------
# Play ground
mat_files = sorted(list(RAW_DIR.glob('*.mat')))
print(mat_files)
mat_file_path = mat_files[0]
for file_path in tqdm(mat_files):
    subject = file_path.stem.split('_')[0]
    epochs = load_epochs_data(file_path)
    epochs.resample(200)
    print(subject, epochs)

    output_file = DATA_DIR.joinpath(
        f'{subject}-freq-CSP-results.joblib')
    if not output_file.is_file():
        try:
            freq_CSP_results = decoding_with_FBCSP_voting(epochs)
            joblib.dump(output_file)
        except:
            pass


# %% ---- 2025-12-01 ------------------------
# Pending


# %% ---- 2025-12-01 ------------------------
# Pending
