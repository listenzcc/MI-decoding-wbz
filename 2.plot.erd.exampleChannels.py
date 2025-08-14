"""
File: 2.plot.erd.exampleChannels.py
Author: Chuncheng Zhang
Date: 2025-08-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot the ERD with example channel(s).

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

# Setup
data_directory = Path('./data/tfr')

output_directory = Path('./data/erd')
output_directory.mkdir(exist_ok=True, parents=True)

compile = re.compile(
    r'^(?P<sn>[A-Za-z0-9]+)-(?P<evt>[A-Za-z0-9]+)-average-tfr.h5')

# %% ---- 2025-08-14 ------------------------
# Function and class


def find_trf_files(src: Path):
    ps = list(data_directory.rglob('*-average-tfr.h5'))
    parsed = [compile.search(p.name).groupdict() for p in ps]
    return [(info, p) for info, p in zip(parsed, ps)]


class PlotOpt:
    # vmin = -0.5
    # vmax = 0.5
    vmin = -10
    vmax = 10
    vcenter = 0
    cmap = 'RdBu'
    channels = ['C3', 'Cz', 'C4']
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    scatter_kwargs = dict(cmap=cmap, marker='s', norm=norm)


# %% ---- 2025-08-14 ------------------------
# Read tfr and generate DataFrame
dfs = []
for info, p in find_trf_files(data_directory):
    print(info)
    print(p)
    tfr = mne.time_frequency.read_tfrs(p)
    print(tfr)
    df = tfr.to_data_frame(long_format=True)
    df['name'] = info['sn']
    df['evt'] = info['evt']
    dfs.append(df)

df = pd.concat(dfs)
df

# %% ---- 2025-08-14 ------------------------
# Plot ERD from DataFrame
evts = sorted(df['evt'].unique())
names = sorted(df['name'].unique())

rows = len(names)
cols = len(PlotOpt.channels)

with PdfPages(output_directory.joinpath('erd.pdf')) as pdf:
    for evt in evts:
        fig_width = 6 * cols  # inch
        fig_height = 4 * rows  # inch
        fig, axes = plt.subplots(
            rows, cols+1,
            figsize=(fig_width, fig_height),
            gridspec_kw={"width_ratios": [10] * cols + [1]})

        for name, chn in tqdm(itertools.product(names, PlotOpt.channels), 'Plotting'):
            i = names.index(name)
            j = PlotOpt.channels.index(chn)
            ax = axes[i, j]

            query = ' & '.join(
                [f'name=="{name}"', f'evt=="{evt}"', f'channel=="{chn}"'])
            _df = df.query(query)

            ax.scatter(_df['time'], _df['freq'],
                       c=_df['value'], **PlotOpt.scatter_kwargs)
            ax.set_title(f'ERD @chn: {chn}, @sub: {name}')
            ax.set_xlabel('Time (s)')
            if j == 0:
                ax.set_ylabel(f'Freq (Hz)')
            else:
                ax.set_yticks([])

        for i in range(rows):
            fig.colorbar(axes[i, 0].collections[0], cax=axes[i, cols],
                         orientation='vertical').ax.set_yscale('linear')

        fig.suptitle(f'{evt=}')
        fig.tight_layout()
        pdf.savefig(fig)
        logger.info(f'Job finished {evt=}')

logger.info(f'Job finished {__file__=}')
# %% ---- 2025-08-14 ------------------------
# Pending
