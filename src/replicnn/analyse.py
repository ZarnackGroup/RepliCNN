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

import string
from sklearn.mixture import GaussianMixture

from .utils.dataio import load_sdf, save_dataframe
from .utils.logger import get_logger
from .utils.misc import moving_average, get_sign_switch_locations, find_contiguous_regions

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)

# make sure there is a logger
logger: logging.Logger = get_logger(level=logging.DEBUG)

# handler for commandline interface
def _analyse(**kwargs) -> None:
	"""Pass arguments to main function and handle output."""

	# load data
	sdf: pd.DataFrame = load_sdf(kwargs["path_sdf"])
	
	# call main function
	output: tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame] = analyse(
		sdf = sdf, 
		smoothing_factor_log2 = int(kwargs["smoothing_factor_log2"]), 
		smoothing_factor_rfd = int(kwargs["smoothing_factor_rfd"]), 
		log = kwargs["log"], 
		)

	# determine experiment name
	experiment_id: str = os.path.splitext(os.path.basename(kwargs["path_sdf"]))[0]

	# handle output
	save_dataframe(dataframe=output[0], path=f"{kwargs['path_out']}{experiment_id}_{kwargs['smoothing_factor_log2']}_{kwargs['smoothing_factor_rfd']}_ori.bed6")
	save_dataframe(dataframe=output[1], path=f"{kwargs['path_out']}{experiment_id}_{kwargs['smoothing_factor_log2']}_{kwargs['smoothing_factor_rfd']}_term.bed6")
	save_dataframe(dataframe=output[2], path=f"{kwargs['path_out']}{experiment_id}_{kwargs['smoothing_factor_log2']}_{kwargs['smoothing_factor_rfd']}_ctr.bed6")
	save_dataframe(dataframe=output[3], path=f"{kwargs['path_out']}{experiment_id}_{kwargs['smoothing_factor_log2']}_{kwargs['smoothing_factor_rfd']}_ttr.bed6")	
	save_dataframe(dataframe=output[4], path=f"{kwargs['path_out']}{experiment_id}_{kwargs['smoothing_factor_log2']}_{kwargs['smoothing_factor_rfd']}_rfd.bed6")

	return None

# main function of this module
def analyse(
	sdf:pd.DataFrame, 
	smoothing_factor_log2:int=3, 
	smoothing_factor_rfd:int=3,
	log:bool=False,  
	logger:logging.Logger=get_logger(level=logging.DEBUG),
	) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
	"""This module takes an sdf-file and calculates origins of replication ,termination zones, timing transition regions, constant timing regions and replication fork directionality."""

	if log: logger.info(f"Smoothing factor for ORI/TERM: {smoothing_factor_log2}")
	if log: logger.info(f"Smoothing factor for CTR/TTR: {smoothing_factor_rfd}")

	# get ORIs
	ori: pd.DataFrame = get_sign_switch_locations(sdf,mode="ori",smoothing_factor=smoothing_factor_log2).reset_index(drop=True).sort_values(by=["chromosome", "start"])

	# get TERMs
	term: pd.DataFrame = get_sign_switch_locations(sdf,mode="term",smoothing_factor=smoothing_factor_log2).reset_index(drop=True).sort_values(by=["chromosome", "start"])

	# get CTRs, TTRs and RFDs	
	# initialise dicts
	sdf_dict: dict = {}
	rfd: dict = {}

	# iterate through all chromosomes
	for chromosome in sdf.chromosome.unique():
		# get data chromosome-wise
		sdf_dict[chromosome]: dict[str,pd.DataFrame] = sdf.query("chromosome==@chromosome").copy()

		# calculate rfd
		rfd[chromosome]: pd.Series = (sdf_dict[chromosome].neg-sdf_dict[chromosome].pos)/(sdf_dict[chromosome].neg+sdf_dict[chromosome].pos)

		# smooth rfd
		rfd[chromosome]: np.ndarray = np.array(pd.Series(moving_average(rfd[chromosome],smoothing_factor_rfd)).round(decimals=3)).reshape(-1, 1)

		# calculate Gaussian mixture model for RFD to separate CTRs from TTRs
		gmm = GaussianMixture(n_components=3,random_state=seed)

		# fit gmm
		gmm.fit(rfd[chromosome])

		# log warning in case the model does not converge
		if not gmm.converged_: logger.warning(f"GaussianMixtureModel did not converge when fitted on RFD of {chromosome}.")

		# predict labels using fitted gmm
		labels = gmm.predict(rfd[chromosome])

		# assign label back to dataframe
		sdf_dict[chromosome]: dict[str,pd.DataFrame] = sdf_dict[chromosome].assign(label=labels)

		# assign rfd back to dataframe
		sdf_dict[chromosome]: dict[str,pd.DataFrame] = sdf_dict[chromosome].assign(rfd=rfd[chromosome])

		# assign TTR and CTR to predicted values
		sorted_indices = np.argsort(gmm.means_.flatten())
		peak_clusters = sorted_indices[[0, 2]]
		flat_cluster = sorted_indices[1]
		label_mapping = {peak_clusters[0]: "TTR", peak_clusters[1]: "TTR", flat_cluster: "CTR"}
		sdf_dict[chromosome]["region_label"] = sdf_dict[chromosome]["label"].map(label_mapping)

	# handle TTRs
	ttr: dict = {chromosome:find_contiguous_regions(sdf_dict[chromosome].query("region_label=='TTR'")) for chromosome in sdf.chromosome.unique()}
	ttr: list = [ttr[chromosome].assign(name=f"TTR_"+chromosome+"_"+(ttr[chromosome].reset_index(drop=True).index+1).astype(str)) for chromosome in sdf.chromosome.unique()]
	ttr: pd.DataFrame = pd.concat(ttr).reset_index(drop=True).assign(score=".").assign(strand=".")[["chromosome","start","end","name","score","strand"]]
	
	# handle CTRs
	ctr: dict = {chromosome:find_contiguous_regions(sdf_dict[chromosome].query("region_label=='CTR'")) for chromosome in sdf.chromosome.unique()}
	ctr: list = [ctr[chromosome].assign(name=f"CTR_"+chromosome+"_"+(ctr[chromosome].reset_index(drop=True).index+1).astype(str)) for chromosome in sdf.chromosome.unique()]
	ctr: pd.DataFrame = pd.concat(ctr).reset_index(drop=True).assign(score=".").assign(strand=".")[["chromosome","start","end","name","score","strand"]]

	# handle RFDs
	rfd: list = [sdf_dict[chromosome].assign(name=f"RFD_"+sdf_dict[chromosome].chromosome+"_"+(sdf_dict[chromosome].reset_index(drop=True).index+1).astype(str)) for chromosome in sdf.chromosome.unique()]
	rfd: pd.DataFrame =  pd.concat(rfd).reset_index(drop=True).assign(strand=".")[["chromosome","start","end","name","rfd","strand"]]

	return (ori, term, ctr, ttr, rfd)