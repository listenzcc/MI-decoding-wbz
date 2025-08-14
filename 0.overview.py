"""
File: 0.overview.py
Author: Chuncheng Zhang
Date: 2025-08-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Overview the data.
    Plot the evoked.
    Plot the sensors.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-08-14 ------------------------
# Requirements and constants
from util.easy_import import *
from util.data import MyData

# Setup
md = MyData()

data_directory = Path('./data/overview')
data_directory.mkdir(parents=True, exist_ok=True)

raw_directory = Path('./raw/20250814-received')
assert raw_directory.is_dir(), f'Dir not exists {raw_directory=}'

# %% ---- 2025-08-14 ------------------------
# Function and class


def find_epochs_files(src: Path):
    '''
    Find epochs files and convert into (path, subject_name) list for output.

    :param path src: The directory to find epochs files in. 

    :return list[tuple[path, str]]: The list of (path, subject_name).
    '''
    # Path list
    ps = list(src.rglob('*_epochs.fif'))
    # Subject name list
    sn = [p.name.replace('_epochs.fif', '') for p in ps]
    return [(p, n) for p, n in zip(ps, sn)]


# %% ---- 2025-08-14 ------------------------
# Play ground
mpl.use('pdf')

for p, subject_name in find_epochs_files(raw_directory):

    # Read data
    md.read_from_file(src=p)
    md.epochs.apply_baseline((None, 0))

    with PdfPages(data_directory.joinpath(f'{subject_name}.pdf')) as pdf:
        # Plot the evoked waveform
        for evt in md.event_id.keys():
            evoked = md.epochs[evt].average()
            fig = evoked.plot(show=False, titles=f'Evoked @{evt=}')
            try:
                pdf.savefig(fig)
            except Exception:
                pass

        # Plot the sensors layout
        fig = evoked.plot_sensors(show_names=True)
        pdf.savefig(fig)

    print(md.epochs.info, file=open(data_directory.joinpath(
        f'{subject_name}-info.txt'), 'w'))

    logger.info(f'Job finished {subject_name=}')

logger.info(f'Job finished {__file__=}')
# %% ---- 2025-08-14 ------------------------
# Pending


# %% ---- 2025-08-14 ------------------------
# Pending
