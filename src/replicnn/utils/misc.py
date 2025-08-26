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
import subprocess

import pandas as pd
import numpy as np
import scipy

import string

from scipy.interpolate import UnivariateSpline

from .logger import get_logger

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)

# make sure there is a logger
logger: logging.Logger = get_logger(level=logging.DEBUG)

def run_command(command:str, log:bool=False) -> None:
	"""Runs the given command."""

	command: str = command + " >/dev/null 2>&1"

	if log: logger.info(f"Running command: {command}")
	return_code: int = os.system(command)
	if return_code!=0: raise Exception(f"Command: {command} failed with error code: {return_code}!")
	
	return None	

def moving_average(x, w):
	"""Calculates the moving average of x with a window of size w."""
	x_new: np.array[float] = np.convolve(x,np.ones(w),"same")/w
	return x_new

def create_tff(condition_1_sdf:list[pd.DataFrame], condition_2_sdf:list[pd.DataFrame], log:bool=False) -> pd.DataFrame:
	"""Takes two lists of sdf-files as pd.DataFrames and creates an tff pd.DataFrame."""

	times_c1: pd.DataFrame = pd.concat([sdf.time for sdf in condition_1_sdf],axis=1)
	times_c2: pd.DataFrame = pd.concat([sdf.time for sdf in condition_2_sdf],axis=1)

	tff: pd.DataFrame = pd.DataFrame({
		"chromosome":condition_1_sdf[0].chromosome,
		"start":condition_1_sdf[0].start,
		"end":condition_1_sdf[0].end,
		"times_c1":times_c1.apply(lambda row: ";".join(row.values.astype(str)), axis=1),
		"mean_c1":times_c1.apply(lambda row: round(np.mean(row),3), axis=1),
		"std_c1":times_c1.apply(lambda row: round(np.std(row),3), axis=1),
		"times_c2":times_c2.apply(lambda row: ";".join(row.values.astype(str)), axis=1),
		"mean_c2":times_c2.apply(lambda row: round(np.mean(row),3), axis=1),
		"std_c2":times_c2.apply(lambda row: round(np.std(row),3), axis=1),
		"p_value":np.nan,
		"p_adj":np.nan,
		"significant":np.nan
	})

	return tff

def create_timing(timing:pd.DataFrame, sdf:pd.DataFrame, chromosomes:pd.Series, log:bool=False) -> np.ndarray[float]:
	"""Takes a sdf-file and a timing file, creates a timing array for supplied sdf."""

	# scale timing to the interval 1 (earliest) and -1 (latest)
	earliest: int = 1
	latest: int = -1
	timing.time: pd.Series[float] = ((timing.time-timing.time.min())/(timing.time.max()-timing.time.min()))*(earliest-latest)+latest

	# initialize empty array for new timings
	new_time: np.ndarray[float] = np.empty(0, dtype="float64")

	for chromosome in chromosomes:
		# subset timing dataframe to current chromosome
		subset_chromosome_time: pd.DataFrame = timing.query("chromosome==@chromosome").reset_index(drop=True)
		subset_chromosome_sdf: pd.DataFrame = sdf.query("chromosome==@chromosome").reset_index(drop=True)
		
		# create a timing spline for each chromosome, take the middle point of the bin for calculating the spline
		spline: UnivariateSpline = UnivariateSpline(x=(((subset_chromosome_time.start+subset_chromosome_time.end))/2).astype(int),
													y=subset_chromosome_time["time"],
													w=None,
													bbox=[None, None],
													k=3,
													s=0,
													ext=0,
													check_finite=False)

		# set bins according to binsize and take the middle point of the bin for evaluation of the spline
		positions: pd.Series[int] = ((subset_chromosome_sdf.start + subset_chromosome_sdf.end)/2).astype(int)

		# evaluate the spline at the given positions and append to array
		new_time: np.ndarray[float] = np.append(new_time, (spline(positions)))

	# cap values at 1 (earliest) and -1 (latest)
	new_time: np.ndarray[float] = np.clip(new_time, -1, 1)

	return new_time

def create_bins(chromsizes:pd.DataFrame, binsize:int, log:bool=False) -> pd.DataFrame:
	"""Takes chromosome sizes and a binsize, creates genome-wide bins."""

	if log: logger.info(f"Creating bins with size {binsize}")
	bins: pd.DataFrame = pd.DataFrame([
		[chrom, start, min(start+binsize,end), f"{chrom}:{start}-{min(start+binsize,end)}"]
		for chrom, start, end in zip(chromsizes["chromosome"], chromsizes["start"], chromsizes["length"])
		for start in range(0,end,binsize)
		])
		
	return bins

def get_free_gpu(log: bool = False) -> int:
    """Return the ID of the GPU with the least memory usage (in MB)."""

	# Get GPU memory usage in MB
	result = subprocess.run(
		"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
		shell=True,
		check=True,
		capture_output=True,
		text=True
	)
	# Parse memory usage values
	mem_used = [int(x) for x in result.stdout.splitlines()]

	if log:
		for idx, mem in enumerate(mem_used):
			print(f"GPU {idx}: {mem} MB used")

	# Return GPU with least memory usage
	return int(np.argmin(mem_used))

def get_sign_switch_locations(sdf:pd.DataFrame, mode:str, smoothing_factor:int=3, log:bool=False) -> pd.DataFrame:
	"""Detects sign switches in a vector. Sign switches are usually indications for replication starting or ending."""

	# define sign change (pos->neg=ORI, neg->pos=TERM)
	if not mode in ["ori","term"]: raise Exception(f"{mode} not in valid modes for sign switch locations.")
	signs = (1,-1) if mode=="ori" else (-1,1)

	# split sdf into chromosomes
	sdf_dict: dict[str,pd.DataFrame] = {chromosome:sdf.query("chromosome==@chromosome") for chromosome in sdf.chromosome.unique()}

	# extract log2 ratio
	sdf_tmp_dict: dict[str,pd.Series] = {chromosome:sdf_dict[chromosome].log2 for chromosome in sdf.chromosome.unique()}

	# smooth log2 ratio
	sdf_tmp_dict: dict[str,pd.Series] = {chromosome:moving_average(sdf_tmp_dict[chromosome],smoothing_factor) for chromosome in sdf.chromosome.unique()}

	# get sign for each element
	sdf_tmp_dict: dict[str,pd.Series] = {chromosome:np.array(np.sign(sdf_tmp_dict[chromosome])) for chromosome in sdf.chromosome.unique()}

	# calculate sign changes
	sdf_tmp_dict: dict[str,pd.Series] = {chromosome:(sdf_tmp_dict[chromosome][:-1]==signs[0])&(sdf_tmp_dict[chromosome][1:]==signs[1]) for chromosome in sdf.chromosome.unique()}

	# get indices of change
	sdf_tmp_dict: dict[str,np.array] = {chromosome:np.where(sdf_tmp_dict[chromosome])[0]+1 for chromosome in sdf.chromosome.unique()}
	
	# find locations
	sdf_tmp_dict: dict[str,pd.DataFrame] = {chromosome:sdf_dict[chromosome].iloc[sdf_tmp_dict[chromosome]].reset_index(drop=True) for chromosome in sdf.chromosome.unique()}

	# add information columns
	sdf_tmp_dict: dict[str,pd.DataFrame] = {chromosome:sdf_tmp_dict[chromosome].assign(name=f"{mode.upper()}_"+sdf_tmp_dict[chromosome].chromosome+"_"+(sdf_tmp_dict[chromosome].index+1).astype(str)).assign(score=".").assign(strand=".") for chromosome in sdf.chromosome.unique()}
	
	# concatenate chromosome-wise data
	locations: pd.DataFrame = pd.concat([sdf_tmp_dict[chromosome] for chromosome in sdf.chromosome.unique()],axis=0).reset_index(drop=True)

	# reformat data
	locations: pd.DataFrame = locations[["chromosome","start","end","name","score","strand"]]

	return locations

def find_contiguous_regions(df:pd.DataFrame) -> pd.DataFrame:
	"""Takes a pandas dataframe and finds contiguous stretches."""

	# copy input dataframe
	df = df.copy() 
	
	# determine binsize
	bin_size = df["start"].diff().dropna().mode()[0]
	
	# identify contiguous regions
	df.loc[:,"group"] = (df["start"]!=df["start"].shift()+bin_size).cumsum()
	
	# group by label and aggregate values
	result = df.groupby(["region_label","group"]).agg(chromosome=("chromosome","first"),start=("start","min"),end=("end","max")).reset_index(drop=True)

	return result

def bed_merge_diff(bed:pd.DataFrame) -> pd.DataFrame:
	"""Takes a bed3, merges consecutive intervals and outputs it as bed6."""

	# determine overlapping regions in the bed file and use group as a helper variable
	bed: pd.DataFrame = bed.assign(group=(bed["chromosome"] != bed["chromosome"].shift()) | (bed["start"] != bed["end"].shift()).cumsum())

	# group the groups and merge entries
	bed: pd.DataFrame = bed.groupby(["chromosome", "group"]).agg(chromosome=("chromosome","first"),start=("start", "first"),end=("end", "last")).reset_index(drop=True)

	# name each interval
	bed: pd.DataFrame = bed.assign(name="DIFF_"+bed["chromosome"]+"_"+(bed.groupby("chromosome").cumcount()+1).astype(str))
	
	# reformat data
	bed: pd.DataFrame = bed.reset_index(drop=True)[["chromosome","start","end","name"]].assign(score=".").assign(strand=".")

	return bed