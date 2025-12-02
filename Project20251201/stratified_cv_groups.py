"""
File: stratified_cv_groups.py
Author: Chuncheng Zhang
Date: 2025-12-01
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Stratified cv groups.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-12-01 ------------------------
# Requirements and constants
import numpy as np
from sklearn.model_selection import StratifiedKFold

# %% ---- 2025-12-01 ------------------------
# Function and class


def stratified_cv_groups(y, n_folds=5):
    """
    生成用于交叉验证的分层分组

    Returns:
    --------
    groups : array
        每个fold的编号 (0到n_folds-1)
    """
    # 使用StratifiedKFold
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
    )

    # 初始化组分配
    groups = np.zeros(len(y), dtype=int)

    # 分配fold编号
    for fold_idx, (_, fold_indices) in enumerate(skf.split(np.zeros(len(y)), y)):
        groups[fold_indices] = fold_idx

    return groups

# %% ---- 2025-12-01 ------------------------
# Play ground


# %% ---- 2025-12-01 ------------------------
# Pending


# %% ---- 2025-12-01 ------------------------
# Pending
