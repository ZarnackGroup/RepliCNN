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
from typing import Dict, List
import logging
import random
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline
import pyBigWig

from .utils.dataio import load_sdf, save_dataframe
from .utils.logger import get_logger
from .utils.ori_ter_helper import create_raw_candidates_rfd_oem, merge_candidates, filter_merged_candidates_by_evidence, recenter_candidates_to_oem_extrema, quantify_ori_term_efficiency_from_bw, filter_efficiency_candidates

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)

# make sure there is a logger
logger: logging.Logger = get_logger(level=logging.DEBUG)

# handler for commandline interface
def _ori_ter(
	input_files: List[str],
	output_prefix: str,
	chrom_sizes_file: str,
	eval_resolution: int,
	save_intermediates: bool = False,
	log: bool = False,
	# Parameters for recenter_candidates_to_oem_extrema
	ori_threshold: float = 0.05,
	ter_threshold: float = 0.15,
	window_radius: int = 15000,
	# Parameters for merge_candidates
	max_merge_size: int = 15000,
	# Parameters for filter_merged_candidates_by_evidence
	n_evidence: int = 2,
	# Parameters for create_raw_candidates_rfd_oem
	smooth_factor_base: float = 1e-3,
	# Parameters for filter_efficiency_candidates
	cutoff: int = 15,
	
) -> None:
	"""
	Wrapper for ori_ter: reads chrom sizes, calls main function, and writes results.
	"""
	logger = get_logger(level=logging.DEBUG if log else logging.INFO)

	# Read chrom sizes file into dict
	chrom_sizes: Dict[str, int] = pd.read_csv(
		chrom_sizes_file, sep="\t", header=None, index_col=0
	).squeeze("columns").to_dict()

	# Call main function
	results = ori_ter(
		input_files=input_files,
		output_prefix=output_prefix,
		chrom_sizes=chrom_sizes,
		eval_resolution=eval_resolution,
		save_intermediates=save_intermediates,
		logger=logger,
		ori_threshold=ori_threshold,
		ter_threshold=ter_threshold,
		window_radius=window_radius,
		max_merge_size=max_merge_size,
		n_evidence=n_evidence,
		smooth_factor_base=smooth_factor_base,
		cutoff=cutoff,
	)

	# Save final filtered_efficiency
	save_dataframe(results["filtered_efficiency"], f"{output_prefix}_oris_ters.bed", log=log)


def ori_ter(
	input_files: List[str],
	output_prefix: str,
	chrom_sizes: Dict[str, int],
	save_intermediates: bool = False,
	eval_resolution: int,
	logger: logging.Logger = None,
	# Parameters for recenter_candidates_to_oem_extrema
	ori_threshold: float = 0.05,
	ter_threshold: float = 0.15,
	window_radius: int = 15000,
	# Parameters for merge_candidates
	max_merge_size: int = 15000,
	# Parameters for filter_merged_candidates_by_evidence
	n_evidence: int = 2,
	# Parameters for create_raw_candidates_rfd_oem
	smooth_factor_base: float = 1e-3,
	# Parameters for filter_efficiency_candidates
	cutoff: int = 15,
) -> Dict[str, pd.DataFrame]:
	"""
	Main ORI/TER detection workflow.
	Takes preloaded RFD/OEM BigWig paths and runs candidate generation,
	merging, filtering, recentering, and efficiency scoring.
	"""
	if logger is None:
		logger = get_logger(level=logging.DEBUG)

	results: Dict[str, pd.DataFrame] = {}

	# Step 1: Generate raw RFD/OEM candidates
	raw_candidates = create_raw_candidates_rfd_oem(
		input_files=input_files,
		out_prefix=output_prefix,
		chrom_sizes=chrom_sizes,
		out_bed=f"{output_prefix}_oris_ters_step1.bed" if save_intermediates else None,
		smooth_factor_base=smooth_factor_base,
	)
	results["raw_candidates"] = raw_candidates

	# Step 2: Merge candidates
	merged_candidates = merge_candidates(
		candidates=raw_candidates,
		max_merge_size=max_merge_size,
		out_file=f"{output_prefix}_oris_ters_step2.bed" if save_intermediates else None,
	)
	results["merged_candidates"] = merged_candidates

	# Step 3: Filter merged candidates by evidence
	filtered_candidates = filter_merged_candidates_by_evidence(
		merged_candidates=merged_candidates,
		n_evidence=n_evidence,
		out_file=f"{output_prefix}_oris_ters_step3.bed" if save_intermediates else None,
	)
	results["filtered_candidates"] = filtered_candidates

	# Step 4: Recenter candidates to OEM extrema
	recentered_candidates = recenter_candidates_to_oem_extrema(
		input_files=input_files,
		candidates=filtered_candidates,
		out_prefix=output_prefix,
		window_radius=window_radius,
		ori_threshold=ori_threshold,
		ter_threshold=ter_threshold,
		eval_resolution=eval_resolution,
		out_file=f"{output_prefix}_oris_ters_step4.bed" if save_intermediates else None,
	)
	results["recentered_candidates"] = recentered_candidates

	# Step 5: Quantify ORI/TER efficiency
	efficiency_scores = quantify_ori_term_efficiency_from_bw(
		input_files=input_files,
		recentered_candidates=recentered_candidates,
		out_prefix=output_prefix,
		eval_resolution=eval_resolution,
		out_file=f"{output_prefix}_oris_ters_step5.bed" if save_intermediates else None,
	)
	results["efficiency_candidates"] = efficiency_scores

	# Step 6: Filter efficiency candidates by cutoff
	filtered_efficiency = filter_efficiency_candidates(
		candidates=efficiency_scores,
		cutoff=cutoff,
		out_file=f"{output_prefix}_oris_ters_step6.bed" if save_intermediates else None,
	)
	results["filtered_efficiency"] = filtered_efficiency

	return results
