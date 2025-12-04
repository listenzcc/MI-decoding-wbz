"""
File: decoding.sliding.py
Author: Chuncheng Zhang
Date: 2025-12-01
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding with Sliding method.

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

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

from mne.decoding import LinearModel, SlidingEstimator, cross_val_multiscore

# %%
DATA_DIR = Path('./data/decoding-sliding')
DATA_DIR.mkdir(exist_ok=True, parents=True)

RAW_DIR = Path('./raw/wbz-20251201-data')
RAW_DIR = Path('./raw/wbz-20251204-data')

# %% ---- 2025-12-01 ------------------------
# Function and class

# %%
# Check results
if True:
    dump_files = DATA_DIR.rglob('*.joblib')
    for path in tqdm(dump_files):
        # scores shape is (n_cv, n_times)
        scores = joblib.load(path)
        avg_scores = np.mean(scores, axis=0)
        plt.plot(avg_scores)
    plt.show()


# %% ---- 2025-12-01 ------------------------
# Play ground
mat_files = sorted(list(RAW_DIR.glob('*.mat')))
print(mat_files)
mat_file_path = mat_files[0]
for file_path in tqdm(mat_files):
    subject = file_path.stem.split('_')[0]
    epochs = load_epochs_data(file_path)
    epochs.resample(200)
    epochs.apply_baseline((None, 0))
    epochs.crop(tmin=0, tmax=4)
    print(subject, epochs)

    X = epochs.get_data()
    y = epochs.events[:, 2]
    cv = 5
    groups = stratified_cv_groups(y)

    # Patterns in sensor space
    clf = make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(solver="liblinear"))
    )

    time_decod = SlidingEstimator(clf, n_jobs=n_jobs, verbose=True)
    # scores shape is (n_cv, n_times)
    scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=n_jobs)
    print(subject, scores)
    joblib.dump(scores, DATA_DIR.joinpath(f'{subject}.scores.joblib'))


# %% ---- 2025-12-01 ------------------------
# Pending


# %% ---- 2025-12-01 ------------------------
# Pending
