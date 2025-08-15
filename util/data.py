"""
File: data.py
Author: Chuncheng Zhang
Date: 2025-08-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    One standing EEG data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-08-14 ------------------------
# Requirements and constants
import mne
from pathlib import Path
from .logging import logger


# %% ---- 2025-08-14 ------------------------
# Function and class
def read_epochs_from_file(src: Path):
    epochs = mne.read_epochs(src)
    return epochs


class MyData(object):
    epochs: mne.Epochs
    events: list
    event_id: dict

    def __init__(self):
        pass

    def read_from_file(self, src: Path):
        epochs = read_epochs_from_file(src)
        logger.info(f'Read epochs from {src=}')

        if epochs.info['sfreq'] > 100:
            epochs.decimate(int(epochs.info['sfreq']/100))

        self.epochs = epochs
        self.events = epochs.events
        self.event_id = epochs.event_id
        return epochs

    def drop_channel(self, ch_name: str):
        '''
        Drop the channel by @ch_name from the epochs.
        The dropping is not case sensitive.

        :param str ch_name: The dropped ch_name.
        '''
        ch_names = [e for e in self.epochs.ch_names
                    if not e.lower() == ch_name.lower()]
        self.epochs.pick_channels(ch_names)


# %% ---- 2025-08-14 ------------------------
# Play ground


# %% ---- 2025-08-14 ------------------------
# Pending


# %% ---- 2025-08-14 ------------------------
# Pending
