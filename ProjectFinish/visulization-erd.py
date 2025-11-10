"""
File: visulization-erd.py
Author: Chuncheng Zhang
Date: 2025-11-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Visualization ERD

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-10 ------------------------
# Requirements and constants
from scipy import signal
from util.easy_import import *

# %%
k_select = 10

n_components = 4
freq_bands = [[4+i*4, 8+i*4] for i in range(9)]+[[8, 32]]
filter_type = 'iir'
filt_order = 5

tmin, tmax = -2, 5
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

# %%
RAW_DIR = Path('./raw/exp_records')

SUBJECT = 'zhangyukun1'

if len(sys.argv) > 2 and sys.argv[1] == '-s':
    SUBJECT = sys.argv[2]

# Every subject has 10 runs
N_RUNS = 10

OUTPUT_DIR = Path(f'./data/exp_record/results/v-erd/{SUBJECT}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2025-11-10 ------------------------
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


def calculate_erd_tfr(epochs, freq_bands=None, baseline=(-2, 0)):
    """
    Calculate ERD using time-frequency representation
    """
    if freq_bands is None:
        freq_bands = {
            'mu': (8, 13),
            'beta': (13, 30)
        }

    from mne.time_frequency import tfr_multitaper

    # Calculate time-frequency representation
    tfr = tfr_multitaper(epochs,
                         freqs=np.arange(1, 35, 1),
                         n_cycles=7,
                         use_fft=True,
                         return_itc=False,
                         average=False,
                         decim=2)

    # Apply baseline correction
    tfr.apply_baseline(baseline=baseline, mode='percent')

    erd_tfr_results = {}

    for band_name, (low_freq, high_freq) in freq_bands.items():
        # Extract data for the frequency band
        freq_mask = (tfr.freqs >= low_freq) & (tfr.freqs <= high_freq)
        band_data = tfr.data[:, :, freq_mask, :].mean(
            axis=2)  # Average over frequencies

        # Average over epochs
        erd_tfr_results[band_name] = np.mean(band_data, axis=0)

    return erd_tfr_results, tfr


# %% ---- 2025-11-10 ------------------------
# Play ground
epochs_all = []

for i in tqdm(range(N_RUNS), f'Loading runs ({SUBJECT=})'):
    X, y = load_data_np(RAW_DIR.joinpath(f'{SUBJECT}/run_{i}.npy'))

    events = np.column_stack((np.array([i*sfreq*8 for i in range(len(y))]),
                              np.zeros(len(y), dtype=int),
                              y))
    epochs = mne.EpochsArray(
        X, info, tmin=tmin, events=events, event_id=event_id)
    epochs_all.append(epochs)

epochs = mne.concatenate_epochs(epochs_all)
print(epochs)

# %% ---- 2025-11-10 ------------------------
# Pending

# First, select the channels of interest
epochs_erd = epochs.copy().pick_channels(['C3', 'Cz', 'C4'])


for evt in ['1', '2']:
    # Calculate ERD using TFR method
    erd_tfr, tfr_obj = calculate_erd_tfr(epochs_erd[evt])

    # Plot ERD results
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Mu rhythm ERD
    for i, channel in enumerate(['C3', 'Cz', 'C4']):
        axes[0].plot(tfr_obj.times, erd_tfr['mu'][i], label=channel)
    axes[0].set_title('Mu Rhythm (8-13 Hz) ERD')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('ERD (%)')
    axes[0].legend()
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Plot Beta rhythm ERD
    for i, channel in enumerate(['C3', 'Cz', 'C4']):
        axes[1].plot(tfr_obj.times, erd_tfr['beta'][i], label=channel)
    axes[1].set_title('Beta Rhythm (13-30 Hz) ERD')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('ERD (%)')
    axes[1].legend()
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

    fig.suptitle(f'{SUBJECT=}, {evt=}')
    plt.tight_layout()
    plt.show()

# Return the ERD results
print("\nSummary of ERD analysis:")
print(f"Channels analyzed: {['C3', 'Cz', 'C4']}")
print(f"Number of epochs: {len(epochs_erd)}")
print(f"Sampling frequency: {sfreq} Hz")

# %% ---- 2025-11-10 ------------------------
# Pending
