"""
File: decode-fbcnet.py
Author: Chuncheng Zhang
Date: 2025-11-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read data and FBCNet it.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-06 ------------------------
# Requirements and constants
import torch
import torch.nn as nn
import torch.optim as optim
from torcheeg.models import FBCNet

from sklearn.metrics import classification_report

from util.easy_import import *
from collect_data import find_bdf_files, read_eeg_data, MyData

# %%
DATA_DIR = Path('./raw/MI-data-2024')
SUBJECT = 'S1'
DEVICE = 3

if len(sys.argv) > 2 and sys.argv[1] == '-s':
    SUBJECT = sys.argv[2]
    DEVICE = int(SUBJECT[1:]) % 6


# %%
FREQ_RANGES = [(e, e+4) for e in range(1, 45, 2)][:200]

# %%
OUTPUT_DIR = Path(f'./data/results/fbcnet/{SUBJECT}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2025-11-06 ------------------------
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


# %% ---- 2025-11-06 ------------------------
# Play ground
table = find_bdf_files(DATA_DIR).query(f'subject == "{SUBJECT}"')
print(table)

mds = []
for i, se in tqdm(table.iterrows(), 'Load data'):
    mds.append(MyData(read_eeg_data(se), se))
print(mds)

epochs = mne.concatenate_epochs([e.epochs for e in mds])
info = epochs.info
X = epochs.get_data()
groups = np.concatenate([np.zeros((len(e.epochs), )) +
                         i for i, e in enumerate(mds)]).ravel()
labels = epochs.events[:, -1]
# groups = np.random.randint(0, 3, (len(labels), ))
print(f'{epochs=}, {groups=}, {labels=}')

# Filter and stack X
baseline = (-1, 0)
tmin, tmax = 0, 4
new_X = []
for low_freq, high_freq in tqdm(FREQ_RANGES, 'Filtering'):
    # 频带滤波
    # X_filtered = X.copy().astype(np.float64, copy=False)
    X_filtered = []

    for i in range(X.shape[0]):
        # 创建Epochs对象进行滤波
        _epochs = mne.EpochsArray(X[i:i+1], info, tmin=epochs.times[0])
        epochs_filtered = _epochs.filter(l_freq=low_freq, h_freq=high_freq,
                                         method='iir', verbose=False)
        epochs_filtered.apply_baseline(baseline)
        epochs_filtered.crop(tmin=tmin, tmax=tmax)
        X_filtered.append(epochs_filtered.get_data()[0])
    new_X.append(X_filtered)

# new_X shape (n_bands, n_samples, n_channels, n_times)
new_X = np.array(new_X)

# Convert into (n_samples, n_bands, n_electrodes, n_times)
X = new_X.transpose((1, 0, 2, 3))
y = labels

print(f'{X.shape=}, {y.shape=}')


# %% ---- 2025-11-06 ------------------------
# Training and testing
y_pred_all = y.copy()
for test_group in np.unique(groups):
    # Make model
    # shape is (n_samples, n_bands, n_electrodes, n_times)
    shape = X.shape
    num_classes = len(np.unique(y))

    # Model
    model = FBCNet(
        num_electrodes=shape[2],
        chunk_size=400,
        in_channels=shape[1],
        num_classes=num_classes,
    ).cuda(DEVICE)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 多分类任务常用
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model, criterion, optimizer)

    # Training loop
    dl = DataLoader(X[:, :, :, :400], y, groups, test_group=test_group)
    it = iter(dl.yield_train_data(batch_size=64))

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

result = {
    'subject': SUBJECT,
    'y_true': y,
    'y_pred': y_pred_all
}
print(classification_report(y_true=y, y_pred=y_pred_all))

joblib.dump(result, OUTPUT_DIR.joinpath('fbcnet-decoding-results.dump'))
logger.info(f'Done with {__file__} > {SUBJECT}')
# %% ---- 2025-11-06 ------------------------
# Pending
