"""
File: collect_data.py
Author: Chuncheng Zhang
Date: 2025-11-04
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Find and read the EEG data.
    The EEG data is paired data.bdf and evt.bdf files.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-04 ------------------------
# Requirements and constants
from util.easy_imports import *

montage = mne.channels.make_standard_montage('standard_1020')

# %% ---- 2025-11-04 ------------------------
# Function and class


def find_bdf_files(root: Path):
    root = Path(root)
    table = []
    for p in sorted(root.rglob('data.bdf')):
        subject = p.parent.parent.name
        name = p.parent.name

        # Make sure the data.bdf and evt.bdf both exist.
        evt = p.with_name('evt.bdf')
        if all([p.is_file(), evt.is_file()]):
            table.append((subject, name, p))

    table = pd.DataFrame(table, columns=['subject', 'name', 'path'])

    return table


def read_eeg_data(record: pd.Series):
    # Read the raw and annotation file
    path = record.path
    raw = mne.io.read_raw_bdf(path)
    ann = mne.read_annotations(path.with_name('evt.bdf'))
    raw.set_annotations(ann)

    # Setup montage
    raw.set_montage(montage, on_missing='warn')

    # Only interested in ['1', '2', '3'] events
    events, event_id = mne.events_from_annotations(raw)
    event_id = {k: v for k, v in event_id.items() if k in ['1', '2', '3']}
    event_nums = event_id.values()
    events = [e for e in events if e[-1] in event_nums]

    # Convert into epochs
    # Crop and down-sample
    decim = int(raw.info['sfreq'] / 200)
    kwargs = {
        'tmin': -2,
        'tmax': 5,
        'decim': decim
    }

    epochs = mne.Epochs(raw, events, event_id, event_repeated='drop', **kwargs)
    ch_names = [e for e in epochs.ch_names if not e[0] in 'EHV']
    epochs.load_data().pick(ch_names)

    return epochs


class MyData(object):
    epochs: mne.Epochs
    finfo: pd.Series

    def __init__(self, epochs, finfo):
        self.epochs = epochs
        self.finfo = finfo
        logger.info(f'Initialized {epochs=}, {finfo=}')

    def __str__(self):
        return f'{self.epochs=}, {self.finfo=}'


# %% ---- 2025-11-04 ------------------------
# Play ground
if __name__ == '__main__':
    table = find_bdf_files('./raw/MI-data-QS')
    print(table)
    se = table.iloc[0]
    epochs = read_eeg_data(se)
    md = MyData(epochs, se)
    print(md)
    print(epochs.ch_names)


# %% ---- 2025-11-04 ------------------------
# Pending


# %% ---- 2025-11-04 ------------------------
# Pending
