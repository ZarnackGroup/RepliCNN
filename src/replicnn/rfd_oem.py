#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RepliCNN - Replication timing prediction and analyses
Copyright (C) 2026 Dominik Stroh

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
from typing import Dict, Tuple
import logging
import random

import pandas as pd
import numpy as np
import pyBigWig

from .utils.dataio import save_dataframe, save_bigwig
from .utils.logger import get_logger

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)

# make sure there is a logger
logger: logging.Logger = get_logger(level=logging.DEBUG)

# handler for commandline interface
def _rfd_oem(
	watson_bw: str,
	crick_bw: str,
	chrom_sizes_file: str,
	resolution: int,
	stride: int,
	output_prefix: str,
	track: str,
	bedgraph: bool = False,
	norm_depth: bool = True,
	invert: bool = True,
) -> None:
	"""
	CLI handler: run rfd_oem and write results to BigWig or bedGraph.

	Parameters
	----------
	watson_bw, crick_bw : str
		Paths to Watson/Crick strand BigWig files.
	chrom_sizes_file : str
		Chromosome sizes file (two columns: chrom, size).
	resolution : int
		Window size (bp).
	stride : int
		Step size (bp).
	output_prefix : str
		Prefix for output file(s).
	track : str
		Either "rfd" or "oem".
	bedgraph : bool, default=False
		If True, write `.bg` (bedGraph), else `.bw` (BigWig).
	norm_depth : bool, default=True
		If True, normalize depth balance.
	invert : bool, default=True
		If True, swap Watson/Crick signals.

	Output
	------
	Writes one file:
	`{output_prefix}_{track}_{stride}_{resolution}.bw` (BigWig)
	or
	`{output_prefix}_{track}_{stride}_{resolution}.bg` (bedGraph).
	"""
	chroms: Dict[str, int] = {}
	with open(chrom_sizes_file) as f:
		for line in f:
			chrom, size_str = line.strip().split()
			chroms[chrom] = int(size_str)

	results = rfd_oem(
		watson_bw=watson_bw,
		crick_bw=crick_bw,
		chroms=chroms,
		resolution=resolution,
		stride=stride,
		track=track,
		norm_depth=norm_depth,
		invert=invert,
	)

	outfile = f"{output_prefix}_{track}_{stride}_{resolution}.{'bg' if bedgraph else 'bw'}"

	if bedgraph:
		all_df = pd.concat(results.values(), ignore_index=True)
		all_df = all_df.sort_values(by=["chrom", "start"])
		save_dataframe(all_df, outfile)
	else:
		save_bigwig(results, chroms, outfile)
		
	return None

def rle_encode_coords(values: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Run-length encode values with respect to genomic coordinates.

	Parameters
	----------
	values : np.ndarray
		Signal values for each position.
	coords : np.ndarray
		Genomic coordinates (same length as values).

	Returns
	-------
	Tuple[np.ndarray, np.ndarray, np.ndarray]
		Arrays of (start, end, value) suitable for BigWig/bedGraph writing.
	"""
	if values.size == 0:
		return np.array([]), np.array([]), np.array([])

	diffs = np.diff(values)
	change_idx = np.where(diffs != 0)[0] + 1
	idx = np.concatenate(([0], change_idx, [len(values)]))

	coords_ext = np.append(coords, coords[-1] + 1)
	starts = coords_ext[idx[:-1]]
	ends = coords_ext[idx[1:]]
	vals = values[idx[:-1]]

	return starts, ends, vals


def compute_rfd(W: np.ndarray, C: np.ndarray, pos: np.ndarray, win: int) -> pd.DataFrame:
	"""
	Compute Replication Fork Directionality (RFD) track for one chromosome.

	Parameters
	----------
	W, C : np.ndarray
		Watson and Crick strand signals.
	pos : np.ndarray
		Positions to compute RFD at.
	win : int
		Window size in base pairs.

	Returns
	-------
	pd.DataFrame
		BedGraph-like DataFrame with columns [start, end, score].
	"""
	half_win = win // 2
	size = len(W)

	W_cum = np.concatenate([[0], np.cumsum(W)])
	C_cum = np.concatenate([[0], np.cumsum(C)])

	starts = np.clip(pos - half_win, 0, size)
	ends = np.clip(pos + half_win, 0, size)

	W_sums = W_cum[ends] - W_cum[starts]
	C_sums = C_cum[ends] - C_cum[starts]
	denom = W_sums + C_sums

	rfd = np.zeros_like(W_sums, dtype=float)
	mask = denom > 0
	rfd[mask] = (W_sums[mask] - C_sums[mask]) / denom[mask]
	rfd = np.round(rfd, 5)

	s, e, v = rle_encode_coords(rfd, pos)
	return pd.DataFrame({"start": s.astype(int), "end": e.astype(int), "score": v.astype(float)})


def compute_oem(W: np.ndarray, C: np.ndarray, pos: np.ndarray, win: int) -> pd.DataFrame:
	"""
	Compute Origin Efficiency Metric (OEM) track for one chromosome.

	Parameters
	----------
	W, C : np.ndarray
		Watson and Crick strand signals.
	pos : np.ndarray
		Positions to compute OEM at.
	win : int
		Window size in base pairs.

	Returns
	-------
	pd.DataFrame
		BedGraph-like DataFrame with columns [start, end, score].
	"""
	size = len(W)

	W_cum = np.concatenate([[0], np.cumsum(W)])
	C_cum = np.concatenate([[0], np.cumsum(C)])

	starts_L = np.clip(pos - win, 0, size)
	ends_L = pos
	W_L = W_cum[ends_L] - W_cum[starts_L]
	C_L = C_cum[ends_L] - C_cum[starts_L]

	starts_R = pos
	ends_R = np.clip(pos + win, 0, size)
	W_R = W_cum[ends_R] - W_cum[starts_R]
	C_R = C_cum[ends_R] - C_cum[starts_R]

	left_frac = W_L / (W_L + C_L + 1e-15)
	right_frac = W_R / (W_R + C_R + 1e-15)
	oem = np.round(left_frac - right_frac, 5) * -1  # invert by convention

	s, e, v = rle_encode_coords(oem, pos)
	return pd.DataFrame({"start": s.astype(int), "end": e.astype(int), "score": v.astype(float)})


def rfd_oem(
	watson_bw: str,
	crick_bw: str,
	chroms: Dict[str, int],
	resolution: int,
	stride: int,
	track: str,
	norm_depth: bool = True,
	invert: bool = True,
) -> Dict[str, pd.DataFrame]:
	"""
	Compute RFD or OEM tracks.

	Parameters
	----------
	watson_bw, crick_bw : str
		Paths to Watson/Crick strand BigWig files.
	chroms : dict
		Dictionary mapping chromosome name -> length.
	resolution : int
		Window size (bp).
	stride : int
		Step size (bp).
	track : str
		Either "rfd" or "oem".
	invert : bool, default=True
		If True, swap Watson/Crick signals.
	norm_depth : bool, default=True
		If True, normalize depth balance.

	Returns
	-------
	Dict[str, pd.DataFrame]
		Dictionary of per-chromosome DataFrames with columns:
		[chrom, start, end, score].
	"""
	results: Dict[str, pd.DataFrame] = {}

	with pyBigWig.open(watson_bw) as w_bw, pyBigWig.open(crick_bw) as c_bw:
		for chrom, size in chroms.items():
			W = np.nan_to_num(w_bw.values(chrom, 0, size, numpy=True))
			C = np.nan_to_num(c_bw.values(chrom, 0, size, numpy=True))
			if invert:
				W, C = C, W

			# normalize depth balance
			if norm_depth:
				W_total, C_total = np.sum(W), np.sum(C)
				ratio = W_total / (C_total + 1e-15)
				if ratio > 1:
					W *= C_total / (W_total + 1e-15)
				elif ratio < 1:
					C *= W_total / (C_total + 1e-15)

			pos = np.arange(0, size, stride)

			if track == "rfd":
				df = compute_rfd(W, C, pos, resolution)
			elif track == "oem":
				df = compute_oem(W, C, pos, resolution)
			else:
				raise ValueError("track must be 'rfd' or 'oem'")

			df.insert(0, "chrom", chrom)
			results[chrom] = df

	return results