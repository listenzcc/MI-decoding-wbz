"""
File: file.py
Author: Chuncheng Zhang
Date: 2025-08-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    File IO with joblib

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-08-11 ------------------------
# Requirements and constants
import joblib
from pathlib import Path


# %% ---- 2025-08-11 ------------------------
# Function and class
def save(value, dst: Path):
    '''
    Save $value into $dst file with joblib.

    :param value: The value to be saved.
    :param Path dst: The file path.
    '''
    dst = Path(dst)
    if dst.is_file():
        import warnings
        warnings.warn(f'File exists: {dst=}')
    joblib.dump(value, dst)
    print(f'Saved file: {dst=}')


def load(src: Path):
    '''
    Load from $src file with joblib.

    :param Path src: The file path.

    :return: The file content.
    '''
    src = Path(src)
    assert src.is_file(), f'File not exists: {src=}'
    print(f'Load file: {src=}')
    return joblib.load(src)


# %% ---- 2025-08-11 ------------------------
# Play ground


# %% ---- 2025-08-11 ------------------------
# Pending


# %% ---- 2025-08-11 ------------------------
# Pending
