"""
File: 1.compute.tfr.py
Author: Chuncheng Zhang
Date: 2025-08-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Compute tfr for the epochs.

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

# Setup
md = MyData()
bands = Bands()

data_directory = Path(f'./data/tfr')
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


def compute_tfr(epochs: mne.Epochs):
    # Setup options
    freqs = np.arange(2, bands.bands['all'][1])  # frequencies start from 2 Hz
    tfr_options = dict(
        method='morlet',
        average=False,
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        n_jobs=n_jobs,
        decim=2
    )
    crop_options = dict(tmin=-2, tmax=epochs.tmax)
    baseline_options = dict(mode='ratio')

    # Load data and compute
    epochs.load_data()
    tfr: mne.time_frequency.EpochsTFR = epochs.compute_tfr(**tfr_options)

    # Crop and apply baseline
    tfr.crop(**crop_options).apply_baseline((None, 0), **baseline_options)

    return tfr


def apply_baseline(tfr: mne.time_frequency.EpochsTFR, mode: str = 'ratio'):
    '''
    Copy the tfr.
    Apply baseline to the tfr with mode.

    mode‘mean’ | ‘ratio’ | ‘logratio’ | ‘percent’ | ‘zscore’ | ‘zlogratio’
    Perform baseline correction by
    - subtracting the mean of baseline values (‘mean’)
    - dividing by the mean of baseline values (‘ratio’)
    - dividing by the mean of baseline values and taking the log (base 10) (‘logratio’)
    - subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
    - subtracting the mean of baseline values and dividing by the standard deviation of baseline values (‘zscore’)
    - dividing by the mean of baseline values, taking the log, and dividing by the standard deviation of log baseline values (‘zlogratio’)
    '''
    tfr = tfr.copy()
    tfr.apply_baseline((None, 0), mode=mode)
    return tfr


# %% ---- 2025-08-14 ------------------------
# Play ground
for p, subject_name in find_epochs_files(raw_directory):
    md.read_from_file(src=p)

    tfr = compute_tfr(md.epochs)
    for evt in md.event_id.keys():
        dst = data_directory.joinpath(f'{subject_name}-{evt}-average-tfr.h5')
        apply_baseline(tfr[evt], mode='logratio').average().save(
            dst, overwrite=True)
        logger.info(f'Saved {dst=}')

    logger.info(f'Job finished {subject_name=}')

logger.info(f'Job finished {__file__=}')

# %% ---- 2025-08-14 ------------------------
# Pending


# %% ---- 2025-08-14 ------------------------
# Pending
