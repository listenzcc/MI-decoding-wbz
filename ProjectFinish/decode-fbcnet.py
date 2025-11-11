"""
File: decode-fbcnet.py
Author: Chuncheng Zhang
Date: 2025-11-07
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decode with FBCNet.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-07 ------------------------
# Requirements and constants
import torch
import torch.nn as nn
import torch.optim as optim
from torcheeg.models import FBCNet
from scipy import signal
from sklearn.model_selection import StratifiedKFold

from util.easy_import import *
from FBCSP.FBCSP_class import filter_bank, FBCSP_info, FBCSP_info_weighted


# %%
RAW_DIR = Path('./raw/exp_records')

SUBJECT = 'zhangyukun1'
DEVICE = np.random.randint(0, 6)

if len(sys.argv) > 2 and sys.argv[1] == '-s':
    SUBJECT = sys.argv[2]

if len(sys.argv) > 4 and sys.argv[3] == '-d':
    DEVICE = int(sys.argv[4])


# Every subject has 10 runs
N_RUNS = 10

OUTPUT_DIR = Path(f'./data/exp_record/results/fbcnet/{SUBJECT}')
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

class DataLoader:
    def __init__(self, X, y, groups, test_group=0):
        self.X = X
        # Scale into 1 scale
        self.X /= np.max(np.abs(self.X))
        self.y = y

        self.X = torch.tensor(self.X).cuda(DEVICE)
        self.y = torch.tensor(self.y).cuda(DEVICE)

        self.groups = groups
        # Separate groups
        unique_groups = sorted(np.unique(self.groups).tolist())
        self.test_groups = [test_group]
        self.train_groups = [
            e for e in unique_groups if not e in self.test_groups]
        logger.info(
            f'DataLoader: {self.X.shape=}, {self.y.shape=}, {self.groups.shape=}, {self.train_groups=}, {self.test_groups=}')

    def yield_train_data(self, batch_size=32):
        train_idx = [g in self.train_groups for g in self.groups]
        while True:
            X = self.X[train_idx]
            y = self.y[train_idx]
            n_samples = X.shape[0]
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                yield X[batch_indices], y[batch_indices]

    def get_test_data(self):
        test_idx = [g in self.test_groups for g in self.groups]
        X = self.X[test_idx]
        y = self.y[test_idx]
        return X, y


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

def mk_groups(X, y):
    cv = StratifiedKFold(n_splits=5)
    cv_list = list(cv.split(X, y))
    groups = y.copy()

    for i, (train_index, test_index) in enumerate(cv_list):
        groups[test_index] = i

    return groups

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

def decode_fbcnet(X, y, groups):
    y_pred_all = y.copy()
    for test_group in np.unique(groups):
        # Make model
        # shape is (n_samples, n_bands, n_electrodes, n_times)
        shape = X.shape
        num_classes = len(np.unique(y))

        # Model
        model = FBCNet(
            num_electrodes=shape[2],
            chunk_size=500,
            in_channels=shape[1],
            num_classes=num_classes,
        ).cuda(DEVICE)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()  # 多分类任务常用
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(model, criterion, optimizer)

        # Training loop
        dl = DataLoader(X[:, :, :, :500], y, groups, test_group=test_group)
        it = iter(dl.yield_train_data(batch_size=10))

        output_path = OUTPUT_DIR.joinpath(f'{test_group}.dump')

        if output_path.is_file():
            continue

        for epoch in tqdm(range(5000), desc='Epoch'):
            def _train():
                X, y = next(it)
                # print(f'{X.shape=}, {y.shape=}')

                _y = model(torch.tensor(X, dtype=torch.float32))
                # print(f'{_y.shape=}')
                # print(_y)

                # 前向传播
                loss = criterion(_y, torch.tensor(y-1))
                # print(f'{loss.item()=}')

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Report
                if epoch % 500 == 0:
                    logger.info(f'Epoch {epoch}, Loss: {loss.item():.6f}')

            _train()

        def _test():
            # Testing loop
            X, y = dl.get_test_data()
            y_true = y.cpu().numpy()
            with torch.no_grad():
                _y = model(torch.tensor(X, dtype=torch.float32)).cpu()
                y_pred = torch.argmax(_y, dim=1).numpy() + 1
                accuracy = np.mean(y_pred == y_true)
                logger.info(f'Test Accuracy ({test_group}): {accuracy * 100:.2f}%')

            y_pred_all[groups == int(test_group)] = y_pred

        _test()

    return y_pred_all

# %% ---- 2025-11-07 ------------------------
# Play ground
acc_all = []
data_all = []
label_all = []
epochs_all = []

for i_run in tqdm(range(N_RUNS), f'Loading runs ({SUBJECT=})'):
    X, y = load_data_np(RAW_DIR.joinpath(f'{SUBJECT}/run_{i_run}.npy'))

    events = np.column_stack((np.array([i*sfreq*8 for i in range(len(y))]),
                              np.zeros(len(y), dtype=int),
                              y))
    epochs = mne.EpochsArray(
        X, info, tmin=tmin, events=events, event_id=event_id)

    groups = mk_groups(X, y)
    print(groups, y)

    new_X = []
    for low_freq, high_freq in tqdm(freq_bands, 'Filtering'):
        # 频带滤波
        X_filtered = []

        for i in range(X.shape[0]):
            # 创建Epochs对象进行滤波
            _epochs = mne.EpochsArray(X[i:i+1], info, tmin=epochs.times[0])
            epochs_filtered = _epochs.filter(l_freq=low_freq, h_freq=high_freq,
                                             method='iir', verbose=False)
            # Downsample to 100 Hz
            epochs_filtered.resample(100)
            # epochs_filtered.crop(tmin=tmin, tmax=tmax)
            X_filtered.append(epochs_filtered.get_data()[0])
        new_X.append(X_filtered)

    # new_X shape (n_bands, n_samples, n_channels, n_times)
    new_X = np.array(new_X)

    # Convert into (n_samples, n_bands, n_electrodes, n_times)
    X = new_X.transpose((1, 0, 2, 3))
    print(X.shape, groups.shape, y.shape)

    decoded = decode_fbcnet(X, y, groups)
    acc_cv = []
    for g in sorted(set(groups)):
        acc = np.sum(y[groups==g]==decoded[groups==g])/len(groups[groups==g])
        acc_cv.append(('fbcnet', acc))
    print(acc_cv)
    joblib.dump(acc_cv, OUTPUT_DIR.joinpath(f'run_{i_run}.dump'))

    continue


# %% ---- 2025-11-07 ------------------------
# Pending


# %% ---- 2025-11-07 ------------------------
# Pending
