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

from .utils.dataio import load_chromsizes, load_bwa, load_bg, save_dataframe
from .utils.logger import get_logger
from .utils.misc import run_command, create_timing, create_bins

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)

# make sure there is a logger
logger: logging.Logger = get_logger(level=logging.DEBUG)

# handler for commandline interface
def _prepare(**kwargs) -> None:
	"""Pass arguments to main function and handle output."""

	# load data
	chromsizes: pd.DataFrame = load_chromsizes(path=kwargs["path_chromsizes"], log=kwargs["log"])
	timing: pd.DataFrame = load_bg(path=kwargs["path_time"], score="time", log=kwargs["log"]) if kwargs["path_time"] else pd.DataFrame()
	
	# call main function
	sdf: pd.DataFrame = prepare(
		path_fwd = kwargs["path_fwd"], 
		path_rev = kwargs["path_rev"], 
		binsize = kwargs["binsize"], 
		chromsizes = chromsizes, 
		timing = timing,
		invert = kwargs["invert"],
		log = kwargs["log"], 
		)

	# handle output
	save_dataframe(dataframe=sdf,path=kwargs["path_out"])

	return None

# main function of this module
def prepare(
	path_fwd:str, 
	path_rev:str, 
	binsize:int, 
	chromsizes:pd.DataFrame, 
	timing:pd.DataFrame=pd.DataFrame(),
	invert:int=False,
	log:bool=False, 
	logger:logging.Logger=get_logger(level=logging.DEBUG)
	) -> pd.DataFrame:
	"""This module takes the user input data and writes a standardised format for this toolbox."""

	# Convert invert to value
	if (log and invert): logger.info("Inverting features.")
	phase = -1 if invert else 1

	# create bed file for bins
	bins = create_bins(chromsizes=chromsizes,binsize=binsize,log=log)

	# write bins to bed4 file
	save_dataframe(dataframe=bins, path="bins.bed4", log=log)

	# run bigWigAverageOverBed
	for strand, path in zip(["pos", "neg"],[path_fwd, path_rev]):
		command: str = f"bigWigAverageOverBed {path} bins.bed4 {strand}.tsv"
		run_command(command=command, log=log)

	# load log2.bg, pos.bg and neg.bg
	pos: pd.DataFrame = load_bwa(path="pos.tsv", score="pos", log=log)
	neg: pd.DataFrame = load_bwa(path="neg.tsv", score="neg", log=log)

	# subset for chosen chromosomes
	pos: pd.DataFrame = pos.query("chromosome in @chromsizes.chromosome").reset_index(drop=True)
	neg: pd.DataFrame = neg.query("chromosome in @chromsizes.chromosome").reset_index(drop=True)

	# order these dataframes to assure same order as in chromsizes
	order: dict = {value: i for i, value in enumerate(chromsizes.chromosome)}
	pos: pd.DataFrame = pos.assign(order=pos.chromosome.map(order)).sort_values(by=["order","start"]).drop(columns="order").reset_index(drop=True)
	neg: pd.DataFrame = neg.assign(order=neg.chromosome.map(order)).sort_values(by=["order","start"]).drop(columns="order").reset_index(drop=True)

	# remove temporary files
	rm_command: str = f"rm ./pos.tsv ./neg.tsv ./bins.bed4"
	run_command(command=rm_command, log=log)

	# initialise arrays to store values
	spline_values: np.ndarray[float] = np.empty(shape=0, dtype="float64")
	derivative_values: np.ndarray[float] = np.empty(shape=0, dtype="float64")
	antiderivative_values: np.ndarray[float] = np.empty(shape=0, dtype="float64")

	# calculate log 2 ratio
	epsilon: float = 0.000000000000001
	log2: pd.Series = np.log2((pos.pos+epsilon)/(neg.neg+epsilon))

	# calculate a spline for each chromosome present in chromsizes
	for chromosome in chromsizes.chromosome:
		# subset to values for the chromosome
		subset_chromosome_log2: pd.Series = log2.iloc[pos.query("chromosome==@chromosome").index]

		spline: scipy.interpolate.UnivariateSpline = scipy.interpolate.UnivariateSpline(
			x=list(range(0,len(subset_chromosome_log2))),
			y=subset_chromosome_log2,
			w=None,
			bbox=[None, None],
			k=3,
			s=0,
			ext=0,
			check_finite=False)

		# calculate the derivative and antiderivative for the spline
		derivative: scipy.interpolate.UnivariateSpline = spline.derivative()
		antiderivative: scipy.interpolate.UnivariateSpline = spline.antiderivative()

		# evaluate the splines at the given positions
		positions: list[int] = list(range(0,len(subset_chromosome_log2)))
		spline_values: np.ndarray[float] = np.append(arr=spline_values, values=scipy.stats.zscore(spline(positions)))
		derivative_values: np.ndarray[float] = np.append(arr=derivative_values, values=scipy.stats.zscore(derivative(positions)))
		antiderivative_values: np.ndarray[float] = np.append(arr=antiderivative_values, values=scipy.stats.zscore(antiderivative(positions)))

	# combine data
	sdf: pd.DataFrame = pd.DataFrame({"chromosome":pos.chromosome, 
									  "start":pos.start, 
									  "end":pos.end, 
									  "pos":pos.pos, 
									  "neg":neg.neg, 
									  "log2":log2 * phase,
									  "spline":spline_values * phase,
									  "derivative":derivative_values * phase, 
									  "antidervative":antiderivative_values * phase, 
									  "time":np.nan})

	# handle timing
	if not timing.empty:
		# get timing
		new_time: np.ndarray[float] = create_timing(timing=timing, sdf=sdf, chromosomes=chromsizes.chromosome)

		# combine timing with sdf
		sdf: pd.DataFrame = sdf.assign(time=new_time)
	
	# round values before returning
	sdf: pd.DataFrame = sdf.round(decimals=3)

	return sdf