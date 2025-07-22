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

from .utils.dataio import load_sdf, load_bg, save_dataframe
from .utils.logger import get_logger
from .utils.misc import create_tff, create_timing, bed_merge_diff
from .utils.stats import test_samples_vs_samples, test_samples_vs_reference, fdr_correction

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)

# handler for commandline interface
def _quantify(**kwargs) -> None:
	"""Pass arguments to main function and handle output."""

	# load data
	condition_1_sdf: list[pd.DataFrame] = [load_sdf(path=path_sdf, log=log) for path_sdf in kwargs["paths_sdf_c1"]]

	if kwargs["paths_sdf_c2"] and not kwargs["path_timing"]:
		with_reference: bool = False
	elif not kwargs["paths_sdf_c2"] and kwargs["path_timing"]:
		with_reference: bool = True
	else:
		logger.error("No second sample group or reference or both were supplied.")
		raise Exception("Supply one of the following: second sample group or reference.")

	# load either reference timing or sample group two
	if with_reference:
		# load reference timing in the used sdf binsize
		condition_2_sdf: list[pd.DataFrame] = [condition_1_sdf[0].assign(time=create_timing(timing=load_bg(path=kwargs["path_time"], score="time", log=kwargs["log"]), sdf=condition_1_sdf[0], chromosomes=condition_1_sdf[0].chromosome.unique()))]
	else:
		# read all sdf-files for condition 2
		condition_2_sdf: list[pd.DataFrame] = [load_sdf(path=path_sdf, log=log) for path_sdf in kwargs["paths_sdf_c2"]]

	# call main function
	result: pd.DataFrame = quantify(
		sdf_c1 = condition_1_sdf,
		sdf_c2 = condition_2_sdf,
		time = time,
		alpha = kwargs["alpha"],
		as_bed = kwargs["as_bed"],
		log = kwargs["log"], 
		)

	# handle output
	save_dataframe(dataframe=tff, path=kwargs["path_out"])

	return result
	
# main function of this module
def quantify(
	sdf_c1:list[pd.DataFrame], 
	sdf_c2:list[pd.DataFrame]=[], 
	time:pd.DataFrame=pd.DataFrame(),
	alpha:float=0.05, 
	as_bed:bool=False,
	log:bool=False, 
	logger:logging.Logger=get_logger(level=logging.DEBUG)
	) -> pd.DataFrame:
	"""This module takes sdf-files from two conditions or from one condition and a reference and calculates significant differences."""

	# determine if user chose to use two sample groups or one sample group and a reference
	if sdf_c2 and time.empty:
		with_reference: bool = False
	elif not sdf_c2 and not time.empty:
		with_reference: bool = True
	else:
		logger.error("No second sample group or reference or both were supplied.")
		raise Exception("Supply one of the following: second sample group or reference.")

	# create tff-file
	tff: pd.DataFrame = create_tff(condition_1_sdf=condition_1_sdf, condition_2_sdf=condition_2_sdf)

	# make timing values accessible
	x_list: pd.Series = tff.times_c1.str.split(";")
	y_list: pd.Series = tff.times_c2.str.split(";")

	# convert timing values to floats
	x_list: pd.Series = x_list.apply(lambda value: list(map(float, value)))
	y_list: pd.Series = y_list.apply(lambda value: list(map(float, value)))

	# deterine which test to use
	if with_reference:
		# use one sample ttest for samples vs reference comparison
		p_values: np.ndarray[float] = np.array([test_samples_vs_reference(x, y) for x, y in zip(x_list, y_list)])
	else:
		# use Welch's ttest for samples vs samples comparison
		p_values: np.ndarray[float] = np.array([test_samples_vs_samples(x, y) for x, y in zip(x_list, y_list)])

	# add p-values to the dataframe
	tff: pd.DataFrame = tff.assign(p_value=p_values)

	# do FDR calculation
	tff: pd.DataFrame = tff.assign(p_adj=fdr_correction(p_values,alpha=alpha))

	# assign if results are significant
	result: pd.DataFrame = tff.assign(significant=np.where(tff.p_adj<alpha,True,False))

	# if referenec is used, round values beofre writing them
	if with_reference:
		result: pd.DataFrame = result.assign(times_c2=result.times_c2.astype(float).round(3))

	# if option is check, convert tff to bed file with significant intervals only
	if as_bed:
		result: pd.DataFrame = bed_merge(result.query("significant==True").reset_index(drop=True)[["chromosome","start","end"]])

	return result