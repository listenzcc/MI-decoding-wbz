"""
File: erd-detection.py
Author: Chuncheng Zhang
Date: 2025-11-05
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Compute and detect ERDs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-05 ------------------------
# Requirements and constants
from matplotlib.colors import TwoSlopeNorm
from mne.stats import permutation_cluster_1samp_test as pcluster_test

from util.easy_import import *
from collect_data import find_bdf_files, read_eeg_data, MyData
from collect_data import find_vhdr_files

# %%
DATA_DIR = Path('./raw/MI-data-2025')
SUBJECT = 'S1'

DATA_DIR = Path('./raw/MI_5')
SUBJECT = 'zzr'

if len(sys.argv) > 2 and sys.argv[1] == '-s':
    SUBJECT = sys.argv[2]

# %% ---- 2025-11-05 ------------------------
# Function and class


# %% ---- 2025-11-05 ------------------------
# Play ground
# table = find_bdf_files(DATA_DIR).query(f'subject == "{SUBJECT}"')
table = find_vhdr_files(DATA_DIR).query(f'subject == "{SUBJECT}"')
print(table)

mds = []
for i, se in tqdm(table.iterrows(), 'Load data'):
    mds.append(MyData(read_eeg_data(se), se))
print(mds)

epochs = mne.concatenate_epochs([e.epochs for e in mds])
print(epochs)
event_ids = list(epochs.event_id.keys())

for evt in event_ids:
    evoked = epochs[evt].average()
    evoked.plot_joint()

epochs.pick(['C3', 'Cz', 'C4'])

# %%
baseline = (-1, 0)  # baseline interval (in s)

freqs = np.arange(2, 36)  # frequencies from 2-35Hz

tfr = epochs.compute_tfr(
    method="morlet",
    freqs=freqs,
    n_cycles=freqs,
    use_fft=True,
    return_itc=False,
    average=False,
    decim=2,
)
tfr.apply_baseline(baseline, mode="logratio")
print(tfr)

# %%
vmin, vmax = -3, 1.5  # set min and max ERDS values in plot
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

for evt in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[evt]
    nc = 3
    fig, axes = plt.subplots(1, nc, figsize=(nc*4, 4))
    for ch, ax in enumerate(axes):  # for each channel
        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot(
            [ch],
            cmap="RdBu",
            # cnorm=cnorm,
            axes=ax,
            colorbar=True,
            show=False,
        )

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.suptitle(f"ERDS ({evt})")
    plt.show()

# %% ---- 2025-11-05 ------------------------
# Pending


# %% ---- 2025-11-05 ------------------------
# Pending
