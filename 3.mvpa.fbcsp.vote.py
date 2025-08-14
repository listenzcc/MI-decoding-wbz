"""
File: 3.mvpa.fbcsp.vote.py
Author: Chuncheng Zhang
Date: 2025-08-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding with FBCSP using proba vote method.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-08-14 ------------------------
# Requirements and constants
from util.easy_import import *
from util.data import MyData
from util.bands import Bands
from util.io.file import save

# Setup
md = MyData()
bands = Bands()

data_directory = Path(f'./data/fbcsp')
data_directory.mkdir(parents=True, exist_ok=True)

raw_directory = Path('./raw/20250814-received')

# %% ---- 2025-08-14 ------------------------
# Function and class


def find_epochs_files(src: Path):
    '''
    Find epochs files and convert into (path, subject_name) list for output.

    :param path src: The directory to find epochs files in. 

    :return list[tuple[path, str]]: The list of (path, subject_name).
    '''
    # Path list
    ps = list(src.rglob('*_epochs.fif'))
    # Subject name list
    sn = [p.name.replace('_epochs.fif', '') for p in ps]
    return [(p, n) for p, n in zip(ps, sn)]


# %% ---- 2025-08-14 ------------------------
# Play ground
freq_ranges = [(e, e+4) for e in range(1, 45, 2)]

for p, subject_name in find_epochs_files(raw_directory):
    md.read_from_file(src=p)

    # Map y: 0 -> 10, 1 -> 11
    y = md.epochs.events[:, 2]
    y += 10

    freq_CSP_results = {
        'subject_name': subject_name,
        'y_true': y,
        'freqs': freq_ranges
    }
    for freqIdx, (fmin, fmax) in enumerate(freq_ranges):
        filter_kwargs = {'l_freq': fmin, 'h_freq': fmax, 'n_jobs': n_jobs}
        epochs_filter = md.epochs.copy()
        epochs_filter.filter(**filter_kwargs)
        epochs_filter.apply_baseline((-2, 0))
        epochs_filter.crop(tmin=0, tmax=4)

        X = epochs_filter.get_data(copy=False)

        from mne.decoding import CSP, Scaler
        from sklearn.metrics import classification_report
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        cv = StratifiedKFold(n_splits=5)
        clf = make_pipeline(
            Scaler(epochs_filter.info),
            CSP(),
            LogisticRegression(),
        )
        y_proba = cross_val_predict(
            estimator=clf, X=X, y=y, cv=cv, method='predict_proba')
        y_pred = np.argmax(y_proba, axis=1) + 10

        print(classification_report(y_true=y, y_pred=y_pred))
        freq_CSP_results[freqIdx] = {
            'fmin': fmin,
            'fmax': fmax,
            'y_proba': y_proba,
            'y_pred': y_pred}

    save(freq_CSP_results, data_directory.joinpath(
        f'{subject_name}-freq-CSP-results.dump'))
    logger.info(f'Job finished {subject_name=}')

logger.info(f'Job finished {__file__=}')

# %% ---- 2025-08-14 ------------------------
# Pending


# %% ---- 2025-08-14 ------------------------
# Pending
