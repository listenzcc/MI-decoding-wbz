"""
File: decode-fbcsp-info.py
Author: Chuncheng Zhang
Date: 2025-11-07
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decode with FBCSP with infomax feature selection.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-07 ------------------------
# Requirements and constants
from scipy import signal
from sklearn.model_selection import StratifiedKFold

from util.easy_import import *
from FBCSP.FBCSP_class import filter_bank, FBCSP_info, FBCSP_info_weighted


# %%

RAW_DIR = Path('./raw/MI-dataset')
SUBJECT = 'sub001'

if len(sys.argv) > 2 and sys.argv[1] == '-s':
    SUBJECT = sys.argv[2]

# Every subject has 10 runs
N_RUNS = 10

OUTPUT_DIR = Path(f'./data/MI-dataset-results/fbcsp-info/{SUBJECT}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
k_select = 10

n_components = 4
freq_bands = [[4+i*4, 8+i*4] for i in range(9)]+[[8, 32]]
filter_type = 'iir'
filt_order = 5

tmin, tmax = 0, 5
sfreq = 250

# 创建info对象
ch_names = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
            'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6']
ch_index = [24, 25, 26, 27, 28, 29, 30,
            34, 35, 36, 37, 38, 39, 40,
            15, 16, 17, 18, 19, 20, 21]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
info.set_montage('standard_1020')
event_id = {
    '1': 1,
    '2': 2
}

# %% ---- 2025-11-07 ------------------------
# Function and class


def load_data_np(path: Path):
    '''
    Read file for EEG data.

    :param path Path: File path of .npy.

    :return X np.array: EEG data (n_samples, n_channels, n_times)
    :return y np.array: EEG label (n_samples, )
    '''

    # Raw data sfreq is 1000 Hz
    raw_sfreq = 1000

    def _load_data(f):
        while True:
            try:
                yield np.load(f)
            except EOFError:
                return

    with open(path, 'rb') as f:
        file_data = np.concatenate(list(_load_data(f))).T

    events = file_data[-1, :]
    file_data = file_data[:-1, :]

    index = []
    for i_point in range(events.shape[0]-1):
        if events[i_point+1] > events[i_point]:
            trial_idx = [int(events[i_point+1]-events[i_point]), i_point+1]
            index.append(trial_idx)

    data_all = []
    label_all = []
    for e in index:
        if e[0] in [1, 2]:
            data_all.append(
                file_data[:, e[1]+int(tmin*raw_sfreq):e[1]+int(tmax*raw_sfreq)])
            label_all.append(e[0])

    data_all = np.array(data_all)[:, ch_index, :]
    X = signal.resample(data_all, sfreq*int(tmax-tmin), axis=-1)
    y = np.array(label_all)
    print(f'{X.shape=}, {y.shape=}, {set(y)=}')
    return X, y


def fbcsp_decoding(X, y):
    '''
    Decoding with FBCSP

    :param X np.array: The data array.
    :param y np.array: The label array.
    '''
    FB = filter_bank(freq_bands, sfreq, filt_order, filter_type)

    cv = StratifiedKFold(n_splits=5)
    cv_list = list(cv.split(X, y))

    acc_cv = []
    for train_index, test_index in tqdm(cv_list, 'CV'):
        fbcsp = FBCSP_info(FB, n_components, (tmin, tmax), k_select)
        fbcsp.fit(X[train_index], y[train_index])

        _pred = fbcsp.predict(X[test_index])
        acc_cv.append(np.mean(y[test_index] == _pred))

    return acc_cv


# %% ---- 2025-11-07 ------------------------
# Play ground
acc_all = []
data_all = []
label_all = []
epochs_all = []

for i in tqdm(range(N_RUNS), f'Loading runs ({SUBJECT=})'):
    X, y = load_data_np(RAW_DIR.joinpath(f'{SUBJECT}/run_{i}.npy'))

    # events = np.column_stack((np.array([i*sfreq*8 for i in range(len(y))]),
    #                           np.zeros(len(y), dtype=int),
    #                           y))
    # epochs = mne.EpochsArray(
    #     X, info, tmin=tmin, events=events, event_id=event_id)

    acc_cv = fbcsp_decoding(X, y)
    acc_all.append(acc_cv)
    print(acc_cv)
    joblib.dump(acc_cv, OUTPUT_DIR.joinpath(f'run_{i}.dump'))
    logger.info(f'Done with {SUBJECT=}, {i=}')

# %% ---- 2025-11-07 ------------------------
# Pending


# %% ---- 2025-11-07 ------------------------
# Pending
