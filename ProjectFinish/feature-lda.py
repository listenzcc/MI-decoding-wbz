"""
File: feature-lda.py
Author: Chuncheng Zhang
Date: 2025-11-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Get feature of LDA and map onto the sensors.

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mne.time_frequency import tfr_multitaper
from mne.viz import plot_topomap

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

OUTPUT_DIR = Path(f'./data/exp_record/results/f-lda/{SUBJECT}')
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
# Select channels of interest (all channels for comprehensive analysis)
epochs_lda = epochs.copy()

# Separate epochs by event type
epochs_1 = epochs_lda['1'].copy()
epochs_2 = epochs_lda['2'].copy()

print(f"Event '1': {len(epochs_1)} epochs")
print(f"Event '2': {len(epochs_2)} epochs")

# Method 1: TFR-based feature extraction for LDA


def extract_tfr_features(epochs, freq_band=(8, 13), time_window=(0, 2)):
    """
    Extract TFR features from epochs for given frequency band and time window
    """
    # Compute time-frequency representation
    tfr = tfr_multitaper(epochs,
                         freqs=np.arange(freq_band[0], freq_band[1] + 1, 1),
                         n_cycles=freq_band[1] - freq_band[0],
                         use_fft=True,
                         return_itc=False,
                         average=False,
                         decim=2)

    # Apply baseline correction if needed
    # tfr.apply_baseline(baseline=(None, 0), mode='percent')

    # Extract data for the specific time window
    times = tfr.times
    time_idx = np.where((times >= time_window[0]) & (
        times <= time_window[1]))[0]

    # Average over frequencies and time points
    # Shape: (n_epochs, n_channels)
    features = np.mean(tfr.data[:, :, :, time_idx], axis=(2, 3))

    return features, tfr.info['ch_names']


# Extract features for both event types
print("Extracting mu band (8-13 Hz) TFR features...")
features_1, ch_names = extract_tfr_features(
    epochs_1, freq_band=(8, 13), time_window=(0, 2))
features_2, _ = extract_tfr_features(
    epochs_2, freq_band=(8, 13), time_window=(0, 2))

print(f"Features shape - Event 1: {features_1.shape}")
print(f"Features shape - Event 2: {features_2.shape}")

# Create labels and combine data
X = np.vstack([features_1, features_2])
y = np.hstack([np.zeros(len(features_1)),  # Label 0 for event '1'
               np.ones(len(features_2))])   # Label 1 for event '2'

print(f"Total samples: {X.shape[0]}")
print(f"Class distribution: {np.unique(y, return_counts=True)}")

# Method 2: Alternative feature extraction - bandpower in time windows


def extract_bandpower_features(epochs, freq_band=(8, 13), sfreq=None):
    """
    Extract bandpower features using Welch's method
    """
    if sfreq is None:
        sfreq = epochs.info['sfreq']

    features = []
    for epoch_data in epochs.get_data():  # epoch_data: (n_channels, n_times)
        epoch_features = []
        for ch_data in epoch_data:
            # Compute PSD using Welch's method
            freqs, psd = signal.welch(
                ch_data, sfreq, nperseg=min(256, len(ch_data)))

            # Extract power in mu band
            band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
            band_power = np.mean(psd[band_mask])
            epoch_features.append(band_power)

        features.append(epoch_features)

    return np.array(features)

# Alternative: Use bandpower features
# features_1_bp = extract_bandpower_features(epochs_1, freq_band=(8, 13), sfreq=epochs.info['sfreq'])
# features_2_bp = extract_bandpower_features(epochs_2, freq_band=(8, 13), sfreq=epochs.info['sfreq'])
# X_bp = np.vstack([features_1_bp, features_2_bp])


# Train LDA classifier
print("\nTraining LDA classifier...")

# Create pipeline with standardization and LDA
pipeline = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
lda = LinearDiscriminantAnalysis()

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

print(
    f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train final model on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lda.fit(X_scaled, y)

# Get LDA weights and coefficients
lda_weights = lda.coef_[0]  # Shape: (n_features,)
intercept = lda.intercept_[0]

print(f"LDA intercept: {intercept:.3f}")
print(f"Number of features: {len(lda_weights)}")

# Create topomap of LDA weights


def plot_lda_topomap(lda_weights, ch_names, info, title="LDA Weights Topomap"):
    """
    Plot topomap of LDA weights
    """
    # Create a mock info structure with only the channels we have
    picks = [info['ch_names'].index(ch)
             for ch in ch_names if ch in info['ch_names']]

    # Create weights array for all channels (NaN for missing channels)
    all_weights = np.full(len(info['ch_names']), np.nan)
    for ch_idx, weight in zip(picks, lda_weights):
        all_weights[ch_idx] = weight

    # Plot topomap
    fig, ax = plt.subplots(figsize=(8, 6))
    im, cm = plot_topomap(all_weights, info, axes=ax, show=False,
                          sensors=True, contours=6, outlines='head')

    ax.set_title(title, fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('LDA Weight', fontsize=12)

    return fig, ax, all_weights


# Plot LDA weights topomap
print("\nCreating LDA weights topomap...")
fig, ax, all_weights = plot_lda_topomap(lda_weights, ch_names, epochs_lda.info,
                                        title="LDA Weights for Mu Band (8-13 Hz) Classification")

# Additional visualization: Compare feature distributions for top channels
# Top 5 most important channels
top_ch_idx = np.argsort(np.abs(lda_weights))[-5:]
top_ch_names = [ch_names[i] for i in top_ch_idx]
top_weights = lda_weights[top_ch_idx]

print(f"\nTop 5 most important channels:")
for ch, weight in zip(top_ch_names, top_weights):
    print(f"  {ch}: {weight:.4f}")

# Plot feature distributions for top channels
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (ch_idx, ch_name) in enumerate(zip(top_ch_idx, top_ch_names)):
    # Features for event 1 and event 2
    features_1_ch = features_1[:, ch_idx]
    features_2_ch = features_2[:, ch_idx]

    axes[i].hist(features_1_ch, alpha=0.7,
                 label="Event '1'", bins=15, density=True)
    axes[i].hist(features_2_ch, alpha=0.7,
                 label="Event '2'", bins=15, density=True)
    axes[i].set_xlabel(f'Mu Band Energy')
    axes[i].set_ylabel('Density')
    axes[i].set_title(f'{ch_name}\nLDA weight: {lda_weights[ch_idx]:.3f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Plot overall classification performance
axes[5].bar(range(len(cv_scores)), cv_scores, color='skyblue', alpha=0.7)
axes[5].axhline(y=cv_scores.mean(), color='red', linestyle='--',
                label=f'Mean: {cv_scores.mean():.3f}')
axes[5].set_xlabel('Cross-validation Fold')
axes[5].set_ylabel('Accuracy')
axes[5].set_title('Cross-validation Performance')
axes[5].legend()
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Method 3: Time-resolved LDA analysis


def time_resolved_lda(epochs_1, epochs_2, freq_band=(8, 13)):
    """
    Perform time-resolved LDA analysis
    """
    # Get common time points
    times = epochs_1.times

    # Initialize arrays for time-resolved analysis
    accuracy_over_time = []
    weights_over_time = []

    # Define time windows for analysis
    window_length = 0.2  # 200 ms windows
    step_size = 0.05     # 50 ms steps

    time_windows = []
    t_start = times[0]

    while t_start + window_length <= times[-1]:
        t_end = t_start + window_length
        time_windows.append((t_start, t_end))
        t_start += step_size

    print(f"\nPerforming time-resolved LDA analysis...")
    print(f"Number of time windows: {len(time_windows)}")

    for i, (start, end) in enumerate(time_windows):
        # Extract features for this time window
        features_1_t, _ = extract_tfr_features(epochs_1, freq_band=freq_band,
                                               time_window=(start, end))
        features_2_t, _ = extract_tfr_features(epochs_2, freq_band=freq_band,
                                               time_window=(start, end))

        X_t = np.vstack([features_1_t, features_2_t])
        y_t = np.hstack([np.zeros(len(features_1_t)),
                        np.ones(len(features_2_t))])

        if len(np.unique(y_t)) < 2:
            continue

        # Cross-validate
        cv_scores_t = cross_val_score(
            pipeline, X_t, y_t, cv=cv, scoring='accuracy')
        accuracy_over_time.append(cv_scores_t.mean())

        # Get weights
        X_t_scaled = scaler.fit_transform(X_t)
        lda_t = LinearDiscriminantAnalysis()
        lda_t.fit(X_t_scaled, y_t)
        weights_over_time.append(lda_t.coef_[0])

    return accuracy_over_time, weights_over_time, time_windows

# Perform time-resolved analysis
# accuracy_over_time, weights_over_time, time_windows = time_resolved_lda(epochs_1, epochs_2)


# Print summary
print("\n" + "="*60)
print("LDA CLASSIFICATION SUMMARY")
print("="*60)
print(f"Event types: '1' vs '2'")
print(f"Frequency band: Mu (8-13 Hz)")
print(f"Number of features: {X.shape[1]} (channels)")
print(
    f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Top discriminative channels: {', '.join(top_ch_names)}")
print(f"LDA model trained successfully!")

# Show the topomap
plt.show()

# %% ---- 2025-11-10 ------------------------
# Pending
