
"""
File: visulization-mrcp.py
Author: Chuncheng Zhang
Date: 2025-11-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Visualization mrcp

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-10 ------------------------
# Requirements and constants
from scipy.stats import ttest_rel
from mne.viz import plot_topomap
from scipy import signal
from datetime import datetime
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
RAW_DIR = Path('./raw/MI-dataset')

SUBJECT = 'sub001'

if len(sys.argv) > 2 and sys.argv[1] == '-s':
    SUBJECT = sys.argv[2]

# Every subject has 10 runs
N_RUNS = 10

OUTPUT_DIR = Path(f'./data/MI-dataset-results/v-mrcp/{SUBJECT}')
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

# Method 1: Direct time-domain analysis for MRCP


def calculate_mrcp_time_domain(epochs, baseline_range=(-2, -1), analysis_range=(-1, 1)):
    """
    Calculate MRCP features in time domain

    Parameters:
    - epochs: epochs object
    - baseline_range: time range for baseline correction (seconds)
    - analysis_range: time range for MRCP analysis (seconds)
    """

    # Apply baseline correction
    epochs_baseline = epochs.copy().apply_baseline(baseline=baseline_range)

    # Get data
    data = epochs_baseline.get_data()  # Shape: (n_epochs, n_channels, n_times)
    times = epochs_baseline.times

    # Define analysis window
    analysis_idx = np.where((times >= analysis_range[0]) & (
        times <= analysis_range[1]))[0]

    mrcp_results = {
        'mean_amplitude': [],
        'min_amplitude': [],
        'time_to_min': [],
        'auc': [],  # Area Under Curve
        'slope': []
    }

    channel_names = epochs.ch_names

    for ch_idx, channel in enumerate(channel_names):
        channel_data = data[:, ch_idx, :]  # Shape: (n_epochs, n_times)

        # Analyze in analysis window
        analysis_data = channel_data[:, analysis_idx]
        analysis_times = times[analysis_idx]

        # Calculate features for each epoch
        mean_amps = []
        min_amps = []
        time_to_mins = []
        aucs = []
        slopes = []

        for epoch_idx in range(len(analysis_data)):
            epoch_signal = analysis_data[epoch_idx]

            # Mean amplitude
            mean_amp = np.mean(epoch_signal)
            mean_amps.append(mean_amp)

            # Minimum amplitude (negative peak)
            min_amp = np.min(epoch_signal)
            min_amps.append(min_amp)

            # Time to minimum
            min_time_idx = np.argmin(epoch_signal)
            time_to_min = analysis_times[min_time_idx]
            time_to_mins.append(time_to_min)

            # Area Under Curve (negative area)
            auc = np.trapz(epoch_signal[epoch_signal < 0],
                           analysis_times[epoch_signal < 0])
            aucs.append(auc)

            # Slope before minimum (from 0 to minimum)
            min_idx = np.argmin(epoch_signal)
            if min_idx > 0:
                slope = (epoch_signal[min_idx] - epoch_signal[0]) / \
                    (analysis_times[min_idx] - analysis_times[0])
            else:
                slope = 0
            slopes.append(slope)

        mrcp_results['mean_amplitude'].append(mean_amps)
        mrcp_results['min_amplitude'].append(min_amps)
        mrcp_results['time_to_min'].append(time_to_mins)
        mrcp_results['auc'].append(aucs)
        mrcp_results['slope'].append(slopes)

    return mrcp_results, times, analysis_times

# Method 2: Low-pass filtering for MRCP analysis


def calculate_mrcp_filtered(epochs, lowpass_freq=5, baseline_range=(-2, -1)):
    """
    Calculate MRCP with low-pass filtering to focus on slow cortical potentials
    """

    # Apply low-pass filter for MRCP (typically < 5 Hz)
    epochs_filtered = epochs.copy().filter(l_freq=None, h_freq=lowpass_freq)

    # Apply baseline correction
    epochs_filtered.apply_baseline(baseline=baseline_range)

    data = epochs_filtered.get_data()
    times = epochs_filtered.times

    # Focus on the period around movement
    movement_range = (-1, 1)  # -1 to +1 seconds around movement
    movement_idx = np.where((times >= movement_range[0]) & (
        times <= movement_range[1]))[0]

    mrcp_features = {}
    channel_names = epochs.ch_names

    for ch_idx, channel in enumerate(channel_names):
        channel_data = data[:, ch_idx, movement_idx]
        movement_times = times[movement_idx]

        # Calculate features
        mean_waveform = np.mean(channel_data, axis=0)
        std_waveform = np.std(channel_data, axis=0)

        # Find negative peak
        neg_peak_idx = np.argmin(mean_waveform)
        neg_peak_time = movement_times[neg_peak_idx]
        neg_peak_amplitude = mean_waveform[neg_peak_idx]

        # Calculate onset latency (when signal drops below -1 µV)
        threshold = -1  # µV
        below_threshold = mean_waveform < threshold
        if np.any(below_threshold):
            onset_idx = np.where(below_threshold)[0][0]
            onset_latency = movement_times[onset_idx]
        else:
            onset_latency = np.nan

        mrcp_features[channel] = {
            'mean_waveform': mean_waveform,
            'std_waveform': std_waveform,
            'neg_peak_amplitude': neg_peak_amplitude,
            'neg_peak_latency': neg_peak_time,
            'onset_latency': onset_latency,
            'movement_times': movement_times
        }

    return mrcp_features


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

# %% ---- 2025-11-10 ------------------------
# Pending
# Select the channels of interest
for evt in ['1', '2']:
    epochs_mrcp = epochs[evt].copy().pick_channels(['C3', 'Cz', 'C4'])

    # Calculate MRCP using both methods
    print("Calculating MRCP features...")

    # Method 1: Time domain analysis
    mrcp_results, all_times, analysis_times = calculate_mrcp_time_domain(
        epochs_mrcp)

    # Method 2: Filtered analysis
    mrcp_filtered = calculate_mrcp_filtered(epochs_mrcp)

    # Print results
    print("\n" + "="*60)
    print("MRCP ANALYSIS RESULTS")
    print("="*60)

    channel_names = epochs_mrcp.ch_names
    print(f"\nTime Domain Features (-1 to +1 seconds):")
    print(f"{'Channel':<8} {'Mean Amp (µV)':<15} {'Min Amp (µV)':<15} {'Time to Min (s)':<15} {'AUC':<15} {'Slope':<15}")
    print("-" * 80)

    for i, channel in enumerate(channel_names):
        mean_amp = np.mean(mrcp_results['mean_amplitude'][i])
        min_amp = np.mean(mrcp_results['min_amplitude'][i])
        time_min = np.mean(mrcp_results['time_to_min'][i])
        auc = np.mean(mrcp_results['auc'][i])
        slope = np.mean(mrcp_results['slope'][i])

        print(f"{channel:<8} {mean_amp:>8.3f} ± {np.std(mrcp_results['mean_amplitude'][i]):.3f}  "
              f"{min_amp:>8.3f} ± {np.std(mrcp_results['min_amplitude'][i]):.3f}  "
              f"{time_min:>8.3f} ± {np.std(mrcp_results['time_to_min'][i]):.3f}  "
              f"{auc:>8.3f} ± {np.std(mrcp_results['auc'][i]):.3f}  "
              f"{slope:>8.3f} ± {np.std(mrcp_results['slope'][i]):.3f}")

    print(f"\nFiltered MRCP Features (low-pass < 5 Hz):")
    print(f"{'Channel':<8} {'Neg Peak (µV)':<15} {'Peak Latency (s)':<15} {'Onset Latency (s)':<15}")
    print("-" * 60)

    for channel in channel_names:
        features = mrcp_filtered[channel]
        print(f"{channel:<8} {features['neg_peak_amplitude']:>8.3f} ± {np.std(features['mean_waveform']):.3f}  "
              f"{features['neg_peak_latency']:>8.3f}          "
              f"{features['onset_latency']:>8.3f}")

    # Plot MRCP results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Grand average MRCP waveforms
    for i, channel in enumerate(channel_names):
        data = epochs_mrcp.copy().apply_baseline(baseline=(-2, -1)).get_data()
        channel_data = data[:, i, :]
        mean_waveform = np.mean(channel_data, axis=0)
        sem_waveform = np.std(channel_data, axis=0) / \
            np.sqrt(len(channel_data))

        axes[0, 0].plot(all_times, mean_waveform, label=channel, linewidth=2)
        axes[0, 0].fill_between(all_times,
                                mean_waveform - sem_waveform,
                                mean_waveform + sem_waveform,
                                alpha=0.3)

    axes[0, 0].axvline(x=0, color='r', linestyle='--',
                       alpha=0.7, label='Movement onset')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (µV)')
    axes[0, 0].set_title('Grand Average MRCP Waveforms')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Filtered MRCP waveforms
    for channel in channel_names:
        features = mrcp_filtered[channel]
        axes[0, 1].plot(features['movement_times'], features['mean_waveform'],
                        label=channel, linewidth=2)
        axes[0, 1].fill_between(features['movement_times'],
                                features['mean_waveform'] -
                                features['std_waveform'],
                                features['mean_waveform'] +
                                features['std_waveform'],
                                alpha=0.3)

    axes[0, 1].axvline(x=0, color='r', linestyle='--',
                       alpha=0.7, label='Movement onset')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude (µV)')
    axes[0, 1].set_title('Low-pass Filtered MRCP (< 5 Hz)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Minimum amplitude distribution
    min_amplitudes = [mrcp_results['min_amplitude'][i]
                      for i in range(len(channel_names))]
    axes[1, 0].boxplot(min_amplitudes, labels=channel_names)
    axes[1, 0].set_ylabel('Minimum Amplitude (µV)')
    axes[1, 0].set_title('Distribution of MRCP Negative Peak Amplitudes')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Topography of MRCP amplitude
    # Average amplitude in the MRCP window (-0.5 to 0.5 s)
    mrcp_window = (-0.5, 0.5)
    window_idx = np.where((all_times >= mrcp_window[0]) & (
        all_times <= mrcp_window[1]))[0]
    window_data = np.mean(data[:, :, window_idx], axis=2)
    mean_amplitude = np.mean(window_data, axis=0)

    # Create a simple topography plot
    # Approximate positions for C3, Cz, C4
    pos = np.array([[-0.05, 0], [0, 0], [0.05, 0]])
    im, cm = plot_topomap(mean_amplitude, pos, axes=axes[1, 1], show=False)
    axes[1, 1].set_title('MRCP Amplitude Topography\n(-0.5 to 0.5 s)')
    plt.colorbar(im, ax=axes[1, 1])

    fig.suptitle(f'{evt=}')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
    plt.show()

    # Statistical analysis
    print(f"\nStatistical Analysis:")
    print(f"{'Comparison':<15} {'t-statistic':<12} {'p-value':<10} {'Significant':<12}")
    print("-" * 50)

    # Compare channels for minimum amplitude
    for i in range(len(channel_names)):
        for j in range(i+1, len(channel_names)):
            t_stat, p_val = ttest_rel(mrcp_results['min_amplitude'][i],
                                      mrcp_results['min_amplitude'][j])
            sig = "Yes" if p_val < 0.05 else "No"
            print(
                f"{channel_names[i]} vs {channel_names[j]:<5} {t_stat:>8.3f}    {p_val:>8.4f}    {sig:>10}")

    print(f"\nMRCP Analysis Summary:")
    print(f"Number of epochs: {len(epochs_mrcp)}")
    print(f"Sampling frequency: {epochs_mrcp.info['sfreq']} Hz")
    print(f"Baseline period: -2 to -1 seconds")
    print(f"Analysis period: -1 to +1 seconds")

# %%
