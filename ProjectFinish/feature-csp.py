"""
File: feature-csp.py
Author: Chuncheng Zhang
Date: 2025-11-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding with csp (in alpha band) and plot the weights as the topomat.

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
from mne.decoding import CSP

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

OUTPUT_DIR = Path(f'./data/MI-dataset-results/f-csp/{SUBJECT}')
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
epochs_csp = epochs.copy()

# Separate epochs by event type
epochs_1 = epochs_csp['1'].copy()
epochs_2 = epochs_csp['2'].copy()

print(f"Event '1': {len(epochs_1)} epochs")
print(f"Event '2': {len(epochs_2)} epochs")

# Method 1: Extract mu band features using bandpass filtering


def extract_mu_band_features(epochs, freq_band=(8, 13), sfreq=None):
    """
    Extract mu band features using bandpass filtering and Hilbert transform
    """
    if sfreq is None:
        sfreq = epochs.info['sfreq']

    # Bandpass filter in mu band
    epochs_filtered = epochs.copy().filter(l_freq=freq_band[0], h_freq=freq_band[1],
                                           method='iir', verbose=False)

    # Get the filtered data
    data = epochs_filtered.get_data()  # Shape: (n_epochs, n_channels, n_times)

    # Compute envelope using Hilbert transform
    analytic_signal = signal.hilbert(data, axis=2)
    envelope = np.abs(analytic_signal)

    # Average over time (you can modify this to use specific time windows)
    features = np.mean(envelope, axis=2)  # Shape: (n_epochs, n_channels)

    return features


# Extract mu band features for both event types
print("Extracting mu band (8-13 Hz) features...")
features_1 = extract_mu_band_features(
    epochs_1, freq_band=(8, 13), sfreq=epochs.info['sfreq'])
features_2 = extract_mu_band_features(
    epochs_2, freq_band=(8, 13), sfreq=epochs.info['sfreq'])

print(f"Features shape - Event 1: {features_1.shape}")
print(f"Features shape - Event 2: {features_2.shape}")

# Prepare data for CSP
X = np.vstack([features_1, features_2])
y = np.hstack([np.zeros(len(features_1)),  # Label 0 for event '1'
               np.ones(len(features_2))])   # Label 1 for event '2'

print(f"Total samples: {X.shape[0]}")
print(f"Class distribution: {np.unique(y, return_counts=True)}")

# Alternative: Use raw data for CSP (more traditional approach)


def prepare_data_for_csp(epochs_1, epochs_2, freq_band=(8, 13)):
    """
    Prepare filtered data for CSP analysis
    """
    # Bandpass filter both epoch sets
    epochs_1_filtered = epochs_1.copy().filter(l_freq=freq_band[0], h_freq=freq_band[1],
                                               method='iir', verbose=False)
    epochs_2_filtered = epochs_2.copy().filter(l_freq=freq_band[0], h_freq=freq_band[1],
                                               method='iir', verbose=False)

    # Get data (keeping time dimension for CSP)
    X1 = epochs_1_filtered.get_data()  # Shape: (n_epochs, n_channels, n_times)
    X2 = epochs_2_filtered.get_data()

    # Combine data
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(len(X1)), np.ones(len(X2))])

    return X, y


# Use traditional CSP approach with time-domain filtered data
print("\nPreparing data for CSP analysis...")
X_csp, y_csp = prepare_data_for_csp(epochs_1, epochs_2, freq_band=(8, 13))
print(f"CSP data shape: {X_csp.shape}")

# Configure CSP
n_components = 4  # Number of CSP components to extract
csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)

# Transform data using CSP
csp_features = csp.fit_transform(X_csp, y_csp)
print(f"CSP features shape: {csp_features.shape}")

# Train LDA on CSP features (standard approach)
lda = LinearDiscriminantAnalysis()
pipeline = make_pipeline(StandardScaler(), lda)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, csp_features,
                            y_csp, cv=cv, scoring='accuracy')

print(
    f"Cross-validation accuracy (CSP + LDA): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train final model
pipeline.fit(csp_features, y_csp)

# Get CSP patterns and filters
csp_patterns = csp.patterns_  # Patterns (source distribution)
csp_filters = csp.filters_    # Spatial filters

print(f"CSP patterns shape: {csp_patterns.shape}")
print(f"CSP filters shape: {csp_filters.shape}")

# Create topomaps for CSP components


def plot_csp_topomaps(patterns, info, component_names=None, figsize=(15, 10)):
    """
    Plot topomaps for CSP patterns
    """
    n_components = patterns.shape[0]

    if component_names is None:
        component_names = [f'CSP Component {i+1}' for i in range(n_components)]

    fig, axes = plt.subplots(2, n_components, figsize=figsize)

    # Plot patterns (source distributions)
    for i in range(n_components):
        im, _ = plot_topomap(patterns[i], info, axes=axes[0, i], show=False,
                             sensors=True, contours=6, outlines='head')
        axes[0, i].set_title(f'{component_names[i]}\nPattern', fontsize=10)
        plt.colorbar(im, ax=axes[0, i], shrink=0.8)

    # Plot filters (spatial filters)
    for i in range(n_components):
        im, _ = plot_topomap(csp_filters[i], info, axes=axes[1, i], show=False,
                             sensors=True, contours=6, outlines='head')
        axes[1, i].set_title(f'{component_names[i]}\nFilter', fontsize=10)
        plt.colorbar(im, ax=axes[1, i], shrink=0.8)

    plt.tight_layout()
    return fig


# Plot CSP patterns and filters
print("\nCreating CSP topomaps...")
fig_topomaps = plot_csp_topomaps(csp_patterns[:n_components], epochs_csp.info,
                                 component_names=[f'Comp {i+1}' for i in range(n_components)])
fig_topomaps.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
plt.show()

# Plot component variances (explained by each CSP component)
fig_var, ax_var = plt.subplots(figsize=(10, 6))
component_vars = np.var(csp_features, axis=0)
total_var = np.sum(component_vars)
explained_var_ratio = component_vars / total_var

ax_var.bar(range(1, n_components + 1), explained_var_ratio * 100,
           color='lightblue', alpha=0.7)
ax_var.set_xlabel('CSP Component')
ax_var.set_ylabel('Explained Variance (%)')
ax_var.set_title('Variance Explained by CSP Components')
ax_var.grid(True, alpha=0.3)

for i, v in enumerate(explained_var_ratio):
    ax_var.text(i + 1, v * 100 + 1, f'{v*100:.1f}%',
                ha='center', va='bottom', fontweight='bold')
fig_var.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
plt.show()

# Plot feature distributions for top CSP components
fig_dist, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i in range(min(4, n_components)):
    comp_features_0 = csp_features[y_csp == 0, i]  # Event '1'
    comp_features_1 = csp_features[y_csp == 1, i]  # Event '2'

    axes[i].hist(comp_features_0, alpha=0.7, label="Event '1'",
                 bins=15, density=True, color='blue')
    axes[i].hist(comp_features_1, alpha=0.7, label="Event '2'",
                 bins=15, density=True, color='red')
    axes[i].set_xlabel(f'CSP Component {i+1} Value')
    axes[i].set_ylabel('Density')
    axes[i].set_title(
        f'CSP Component {i+1}\nVariance: {explained_var_ratio[i]*100:.1f}%')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Plot cross-validation performance
if len(axes) > 4:
    axes[4].bar(range(len(cv_scores)), cv_scores, color='skyblue', alpha=0.7)
    axes[4].axhline(y=cv_scores.mean(), color='red', linestyle='--',
                    label=f'Mean: {cv_scores.mean():.3f}')
    axes[4].set_xlabel('Cross-validation Fold')
    axes[4].set_ylabel('Accuracy')
    axes[4].set_title('CSP + LDA Cross-validation Performance')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

plt.tight_layout()
fig_dist.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
plt.show()

# Create a summary topomap showing the most discriminative patterns


def plot_csp_discriminative_patterns(patterns, info, n_components=4):
    """
    Plot the most discriminative CSP patterns
    """
    fig, axes = plt.subplots(1, n_components, figsize=(16, 4))

    for i in range(n_components):
        # Normalize pattern for better visualization
        pattern_normalized = patterns[i] / np.max(np.abs(patterns[i]))

        im, _ = plot_topomap(pattern_normalized, info, axes=axes[i], show=False,
                             sensors=True, contours=6, outlines='head')
        axes[i].set_title(
            f'CSP Pattern {i+1}\n(Most Discriminative)', fontsize=12)
        plt.colorbar(im, ax=axes[i], shrink=0.8)

    plt.tight_layout()
    return fig


# Plot discriminative patterns
fig_discrim = plot_csp_discriminative_patterns(
    csp_patterns[:n_components], epochs_csp.info)
fig_discrim.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))

# Print component analysis
print("\n" + "="*60)
print("CSP COMPONENT ANALYSIS")
print("="*60)

for i in range(n_components):
    comp_features_0 = csp_features[y_csp == 0, i]
    comp_features_1 = csp_features[y_csp == 1, i]

    mean_0, mean_1 = np.mean(comp_features_0), np.mean(comp_features_1)
    std_0, std_1 = np.std(comp_features_0), np.std(comp_features_1)

    # Calculate discriminative power (absolute mean difference normalized by pooled std)
    pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)
    discriminative_power = np.abs(mean_1 - mean_0) / pooled_std

    print(f"Component {i+1}:")
    print(f"  Event '1' mean: {mean_0:.3f} ± {std_0:.3f}")
    print(f"  Event '2' mean: {mean_1:.3f} ± {std_1:.3f}")
    print(f"  Discriminative power: {discriminative_power:.3f}")
    print(f"  Variance explained: {explained_var_ratio[i]*100:.1f}%")
    print()

# Print summary
print("\n" + "="*60)
print("CSP CLASSIFICATION SUMMARY")
print("="*60)
print(f"Event types: '1' vs '2'")
print(f"Frequency band: Mu (8-13 Hz)")
print(f"Number of CSP components: {n_components}")
print(
    f"Cross-validation accuracy (CSP + LDA): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Most discriminative component: {np.argmax(explained_var_ratio) + 1}")

# Show all plots
plt.show()

# %% ---- 2025-11-10 ------------------------
# Pending
