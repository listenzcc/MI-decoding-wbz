"""
File: read_data.py
Author: Chuncheng Zhang
Date: 2025-12-01
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read data from .mat files.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-12-01 ------------------------
# Requirements and constants
import scipy.io as sio
from util.easy_import import *

RAW_DIR = Path('./raw/wbz-20251201-data')

# %% ---- 2025-12-01 ------------------------
# Function and class


def detect_bad_channels(X):
    X = X.copy()
    x = X.transpose(2, 0, 1).reshape((32, -1))
    s = np.std(x, axis=1)
    bad_index = np.where(s == 0)[0]
    return bad_index


def load_epochs_data(mat_file_path):
    obj = sio.loadmat(mat_file_path, squeeze_me=True)
    data = {}
    for k, v in obj.items():
        if k.startswith('__'):
            continue
        try:
            print(k, v.shape)
        except:
            print(k, v)

    # ! Currently data shape is (n_epochs, n_times, n_channels)
    data = {
        'data': obj['epochdata'].squeeze(),
        'task': obj['task_seq'].squeeze(),
        'tmin': -2,
        'tmax': 7,
        'ttask': (0, 4)
    }

    n_epochs, n_times, n_channels = data['data'].shape

    bad_index = detect_bad_channels(data['data'])
    X_map = [e for e in range(32) if e not in bad_index]

    # ! Transpose to (n_epochs, n_channels, n_times)
    data['data'] = data['data'].transpose([0, 2, 1])
    data['data'] = data['data'][:, X_map, :]

    n_channels = data['data'].shape[1]

    assert n_epochs == len(data['task']), 'Incorrect epochsdata or task_seq'
    events = np.column_stack([
        np.arange(0, n_epochs * 3000, 3000),  # 事件发生样本点（假设每试次间隔3000个样本）
        np.zeros(n_epochs, dtype=int),
        data['task'].astype(int)
    ])
    event_id = {'1': 1, '2': 2}

    # Make info
    info = mne.create_info(n_channels, sfreq=2000, ch_types='eeg')

    epochs = mne.EpochsArray(
        data['data'], info, events=events, event_id=event_id, tmin=data['tmin'])

    return epochs


# %% ---- 2025-12-01 ------------------------
# Play ground
if __name__ == '__main__':
    mat_files = sorted(list(RAW_DIR.glob('*.mat')))
    display(mat_files)

    epochs = load_epochs_data(mat_files[0])
    print(epochs)

# %% ---- 2025-12-01 ------------------------
# Pending


# %% ---- 2025-12-01 ------------------------
# Pending
