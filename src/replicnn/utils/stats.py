#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RepliCNN - Replication timing prediction and analyses
Copyright (C) 2025 Dominik Stroh

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/gpl-3.0.html
"""

# set restriction
if __name__=="__main__":
	raise RuntimeError("This script cannot be run directly. Please import it as a module.")

# import packages
import os
import shutil
import sys
import typing
import logging
import random

import pandas as pd
import numpy as np
import scipy

from statsmodels.stats.multitest import multipletests

from .logger import get_logger

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)

# make sure there is a logger
logger: logging.Logger = get_logger(level=logging.DEBUG)

def test_samples_vs_samples(group_1:list[float], group_2:list[float]) -> float:
	"""Compare two samples groups with eachother (Welch's ttest). Assume normality and not equal variances."""
	
	# test the samples
	p_value: float = scipy.stats.ttest_ind(group_1, group_2, equal_var=False, alternative="two-sided", axis=0, nan_policy="propagate", permutations=None, random_state=seed, trim=0)[1]

	return p_value

def test_samples_vs_reference(group_1,reference) -> float:
	"""Compare one sample group to a reference (one sample ttest). Assume normality, variance can be ignored."""

	# test the samples
	p_value: float = scipy.stats.ttest_1samp(group_1, reference, alternative="two-sided", axis=0, nan_policy="propagate")[1]

	return p_value

def fdr_correction(p_values:np.ndarray[float], alpha:float=0.05, method:str="fdr_bh", log:bool=False) -> np.ndarray[float]:
	"""Take a set of p-values and do a multiple testing correction."""

	# handle nans
	mask: pd.Series[bool] = np.isfinite(p_values)
	p_adj: np.ndarray[float] = np.empty(p_values.shape)
	p_adj.fill(np.nan)

	# do multiple testing correction
	p_adj[mask]: np.ndarray[float] = multipletests(p_values[mask], alpha=alpha, method=method)[1]

	return p_adj