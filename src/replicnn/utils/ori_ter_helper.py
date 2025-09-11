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
from typing import Dict, List, Tuple, Optional
import logging
import random

import pandas as pd
import numpy as np
import scipy
import re

os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras

from .logger import get_logger

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)

# make sure there is a logger
logger: logging.Logger = get_logger(level=logging.DEBUG)

def create_raw_candidates_rfd_oem(
	input_files: List[str],
	out_prefix: str,
	chrom_sizes: Dict[str, int],
	out_bed: Optional[str] = None,
	smooth_factor_base: float = 1e-3,
) -> Dict[str, List[Tuple[str, int, int, str, float, float, int]]]:
	"""
	Identify raw replication origin (ORI) and termination (TER) candidates
	using both RFD (replication fork directionality) and OEM (origin efficiency metric).

	Input filenames must follow the convention:
		id_{rfd|oem}_{stride}_{resolution}.bw

	Candidate tuple format (in memory):
		(chrom, start, end, name, rfd_score, oem_score, resolution)

	File output (if out_bed is set):
		BED4 -> chrom, start, end, name
	"""

	# Parse metadata from filenames
	file_info = []
	for f in input_files:
		fname = os.path.basename(f)
		match = re.match(r".*_(rfd|oem)_(\d+)_(\d+)\.bw$", fname)
		if not match:
			raise ValueError(f"Input file name does not match pattern: {fname}")
		track, stride, resolution = match.groups()
		file_info.append({
			"path": f,
			"track": track.lower(),
			"stride": int(stride),
			"resolution": int(resolution),
		})

	raw_candidates: List[Tuple[str, int, int, str, float, float, int]] = []
	candidates: Dict[str, List[Tuple[str, int, int, str, float, float, int]]] = {chrom: [] for chrom in chrom_sizes}

	# Separate RFD and OEM files
	rfd_files = [f for f in file_info if f["track"] == "rfd"]
	oem_files = [f for f in file_info if f["track"] == "oem"]

	# --- helper: find RFD candidates ---
	def find_rfd_candidates(chrom, pos, rfd, oem_dict, win, stride) -> List[Tuple]:
		cands: List[Tuple] = []
		rfd = np.nan_to_num(rfd)

		if len(pos) < 5:
			return cands

		smooth = smooth_factor_base * len(pos)
		spline = UnivariateSpline(pos, rfd, s=smooth)
		x_fine = np.arange(pos[0], pos[-1], 1)
		rfd_fine = spline(x_fine)

		signs = np.sign(rfd_fine)
		switch_idx = np.where(np.diff(signs) != 0)[0]

		for i in switch_idx:
			rfd_val = float(rfd_fine[i])
			coord = x_fine[i]

			# Map coordinate into OEM values
			oem_vals = []
			for w, arr in oem_dict.items():
				idx = coord // stride
				if idx < len(arr):
					oem_vals.append(arr[int(idx)])
			if not oem_vals:
				continue

			oem_val = float(np.mean(oem_vals))
			name = f"{'ORI' if rfd_val < 0 else 'TER'}_RFD_{win}"
			cand = (chrom, int(coord), int(coord) + 1, name, rfd_val, oem_val, win)
			raw_candidates.append(cand)
			cands.append(cand)

		return cands

	# --- helper: find OEM candidates ---
	def find_oem_candidates(chrom, pos, oem, win) -> List[Tuple]:
		cands: List[Tuple] = []
		oem = np.nan_to_num(oem)

		smooth = smooth_factor_base * len(pos)
		spline = UnivariateSpline(pos, oem, s=smooth)
		x_fine = np.arange(pos[0], pos[-1], 1)
		oem_fine = spline(x_fine)

		maxima = argrelextrema(oem_fine, np.greater)[0]
		minima = argrelextrema(oem_fine, np.less)[0]

		for i in maxima:
			val = float(oem_fine[i])
			if val <= 0:
				continue
			coord = x_fine[i]
			name = f"ORI_OEM_{win}"
			cand = (chrom, int(coord), int(coord) + 1, name, 0.0, val, win)
			raw_candidates.append(cand)
			cands.append(cand)

		for i in minima:
			val = float(oem_fine[i])
			if val >= 0:
				continue
			coord = x_fine[i]
			name = f"TER_OEM_{win}"
			cand = (chrom, int(coord), int(coord) + 1, name, 0.0, val, win)
			raw_candidates.append(cand)
			cands.append(cand)

		return cands

	# --- preload OEM signals ---
	oem_data: Dict[int, Dict[str, np.ndarray]] = {}
	for f in oem_files:
		win = f["resolution"]
		stride = f["stride"]
		oem_data[win] = {}
		with pyBigWig.open(f["path"]) as bw:
			for chrom, size in chrom_sizes.items():
				oem_vals = np.nan_to_num(bw.values(chrom, 0, size, numpy=True))[::stride]
				oem_data[win][chrom] = oem_vals

	# --- collect candidates per RFD file ---
	for f in rfd_files:
		win = f["resolution"]
		stride = f["stride"]
		with pyBigWig.open(f["path"]) as bw:
			for chrom, size in chrom_sizes.items():
				pos = np.arange(0, size, stride)
				rfd_vals = np.nan_to_num(bw.values(chrom, 0, size, numpy=True))[::stride]

				# Use OEMs for same chrom
				oem_dict = {w: arr[chrom] for w, arr in oem_data.items() if chrom in arr}

				candidates[chrom].extend(find_rfd_candidates(chrom, pos, rfd_vals, oem_dict, win, stride))

	# --- collect OEM-only candidates ---
	for f in oem_files:
		win = f["resolution"]
		stride = f["stride"]
		with pyBigWig.open(f["path"]) as bw:
			for chrom, size in chrom_sizes.items():
				pos = np.arange(0, size, stride)
				oem_vals = np.nan_to_num(bw.values(chrom, 0, size, numpy=True))[::stride]
				candidates[chrom].extend(find_oem_candidates(chrom, pos, oem_vals, win))

	# --- save raw candidates (BED4 only) ---
	if out_bed:
		with open(out_bed, "w") as out:
			for chrom, start, end, name, *_ in raw_candidates:
				strand = "+" if name.startswith("ORI") else "-"
				out.write(f"{chrom}\t{start}\t{end}\t{name}\t.\t{strand}\n")

	return candidates

def merge_candidates(
	candidates: Dict[str, List[Tuple[str, int, int, str, float, float, int]]],
	max_merge_size: int = 2500,
	out_file: Optional[str] = None,
) -> Dict[str, List[Tuple[int, int, int, str, float, float, List[int], List[str]]]]:
	"""
	Merge ORI and TER candidates separately based on max_merge_size.
	Optionally write merged candidates to a BED6 file.

	Input
	-----
	candidates : dict
		{chrom: [(chrom, start, end, name, rfd_val, oem_val, win), ...]}
	max_merge_size : int
		Maximum distance between consecutive candidates to merge them.
	out_file : str or None
		If provided, writes merged candidates in BED6 format:
		chrom, start, end, name, score (.), strand (+ for ORI, - for TER).

	Output
	------
	merged : dict
		{chrom: [(start, end, center, type, mean_rfd, mean_oem, wins, types), ...]}
	"""
	merged: Dict[str, List[Tuple[int, int, int, str, float, float, List[int], List[str]]]] = defaultdict(list)
	lines_to_write: List[str] = []

	for chrom, cand_list in candidates.items():
		# separate ORIs and TERs
		oris = [c for c in cand_list if c[3].startswith("ORI")]
		ters = [c for c in cand_list if c[3].startswith("TER")]

		for group in [oris, ters]:
			if not group:
				continue

			# sort by genomic start
			group = sorted(group, key=lambda x: x[1])

			# initialize first cluster
			curr_start, curr_end = group[0][1], group[0][2]
			rfd_vals = [group[0][4]]
			oem_vals = [group[0][5]]
			wins = [group[0][6]]
			types = [group[0][3]]

			for c in group[1:]:
				s, e = c[1], c[2]
				if s - curr_end <= max_merge_size:
					# extend current cluster
					curr_end = max(curr_end, e)
					rfd_vals.append(c[4])
					oem_vals.append(c[5])
					wins.append(c[6])
					types.append(c[3])
				else:
					# finalize current cluster
					center = (curr_start + curr_end) // 2
					cluster_type = types[0]  # ORI or TER
					merged[chrom].append((
						curr_start, curr_end, center,
						cluster_type,
						float(np.mean(rfd_vals)),
						float(np.mean(oem_vals)),
						wins.copy(),
						types.copy()
					))
					if out_file:
						strand = "+" if cluster_type.startswith("ORI") else "-"
						lines_to_write.append(
							f"{chrom}\t{curr_start}\t{curr_end}\t{cluster_type}\t.\t{strand}\n"
						)
					# reset cluster
					curr_start, curr_end = s, e
					rfd_vals = [c[4]]
					oem_vals = [c[5]]
					wins = [c[6]]
					types = [c[3]]

			# flush final cluster
			center = (curr_start + curr_end) // 2
			cluster_type = types[0]
			merged[chrom].append((
				curr_start, curr_end, center,
				cluster_type,
				float(np.mean(rfd_vals)),
				float(np.mean(oem_vals)),
				wins.copy(),
				types.copy()
			))
			if out_file:
				strand = "+" if cluster_type.startswith("ORI") else "-"
				lines_to_write.append(
					f"{chrom}\t{curr_start}\t{curr_end}\t{cluster_type}\t.\t{strand}\n"
				)

	if out_file:
		with open(out_file, "w") as f:
			f.writelines(lines_to_write)

	return merged


def filter_merged_candidates_by_evidence(
	merged_candidates: Dict[str, List[Tuple[int, int, int, str, float, float, List[int], List[str]]]],
	n_evidence: int = 2,
	out_file: Optional[str] = None,
) -> Dict[str, List[Tuple[int, int, int, str, float, float, List[int], List[str]]]]:
	"""
	Filter merged candidates based on RFD and OEM evidence.

	Parameters
	----------
	merged_candidates : dict
		Output of merge_candidates():
		{chrom: [(start, end, center, type, mean_rfd, mean_oem, wins, types), ...]}
	n_evidence : int
		Minimum total supporting signals (RFD + OEM) required to keep a candidate.
	out_file : str or None
		If provided, writes filtered candidates to this file (BED6).

	Returns
	-------
	filtered : dict
		{chrom: [(start, end, center, type, mean_rfd, mean_oem, wins, types), ...]}
	"""
	filtered: Dict[str, List[Tuple[int, int, int, str, float, float, List[int], List[str]]]] = defaultdict(list)
	lines_to_write: List[str] = []

	for chrom, clusters in merged_candidates.items():
		for start, end, center, name, mean_rfd, mean_oem, wins, types in clusters:
			# count evidence types
			n_rfd = sum("RFD" in t for t in types)
			n_oem = sum("OEM" in t for t in types)

			if n_rfd + n_oem >= n_evidence:
				filtered[chrom].append((start, end, center, name, mean_rfd, mean_oem, wins, types))
				if out_file:
					strand = "+" if name.startswith("ORI") else "-"
					# BED6: chrom, start, end, name, score, strand
					lines_to_write.append(
						f"{chrom}\t{start}\t{end}\t{name}\t.\t{strand}\n"
					)

	if out_file:
		with open(out_file, "w") as f:
			f.writelines(lines_to_write)

	return filtered

def recenter_candidates_to_oem_extrema(
    candidates: Dict[str, List[Tuple[int, int, int, str, float, float, List[int], List[str]]]],
    input_files: List[str],
    out_prefix: str,
    eval_resolution: int,
    window_radius: int = 5000,
    ori_threshold: float = 0.5,
    ter_threshold: float = 0.5,
    out_file: Optional[str] = None,
) -> List[Tuple[str, int, int, str, int]]:
    """
    Recenters ORI/TER candidates to the nearest local max/min of the OEM signal
    using the OEM BigWig file that matches eval_resolution.

    Parameters
    ----------
    candidates : dict
        Dictionary of candidates per chromosome.
    input_files : list of str
        OEM BigWig file paths, following pattern: id_oem_stride_resolution.bw
    out_prefix : str
        Prefix for output BED file if out_file is None.
    eval_resolution : int
        Resolution (bp) used to select the correct OEM BigWig file.
    window_radius : int
        Search radius around candidate center (bp).
    ori_threshold : float
        Fraction (0–1). ORI boundaries are set where OEM < (1 - ori_threshold) * peak_value.
    ter_threshold : float
        Fraction (0–1). TER boundaries are set where OEM > (1 - ter_threshold) * dip_value.
    out_file : str, optional
        BED6 output file path.

    Returns
    -------
    recentered : list of tuples
        [(chrom, new_start, new_end, name, new_center), ...]
    """
    if out_file is None:
        out_file = f"{out_prefix}_recentered_candidates.bed"

    # Select OEM file with matching resolution
    oem_files = [
        f for f in input_files
        if re.search(rf"_oem_\d+_{eval_resolution}\.bw$", os.path.basename(f), re.IGNORECASE)
    ]
    if not oem_files:
        raise ValueError(f"No OEM BigWig file found for resolution={eval_resolution} in input_files.")
    oem_file = oem_files[0]

    # Parse stride from filename
    match = re.search(r"_oem_(\d+)_(\d+)\.bw$", os.path.basename(oem_file), re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse stride/resolution from OEM file: {oem_file}")
    stride = int(match.group(1))
    resolution = int(match.group(2))
    assert resolution == eval_resolution, f"File resolution {resolution} ≠ eval_resolution {eval_resolution}"

    recentered: List[Tuple[str, int, int, str, int]] = []

    # Open BigWig safely
    with pyBigWig.open(oem_file) as oem_bw:
        for chrom, cand_list in candidates.items():
            chrom_size = oem_bw.chroms()[chrom]
            per_base_signal = np.nan_to_num(oem_bw.values(chrom, 0, chrom_size, numpy=True))
            signal = per_base_signal[::stride]

            for start, end, center, name, rfd_val, oem_val, wins, types in cand_list:
                center_bin = center // stride
                left_bin = max(0, (center - window_radius) // stride)
                right_bin = min(signal.size, (center + window_radius) // stride)
                local_signal = signal[left_bin:right_bin]

                if local_signal.size == 0:
                    continue

                maxima = argrelextrema(local_signal, np.greater)[0]
                minima = argrelextrema(local_signal, np.less)[0]

                if "ORI" in name and maxima.size > 0:
                    rel_idx = np.argmin(np.abs(maxima - (center_bin - left_bin)))
                    new_bin = maxima[rel_idx] + left_bin
                    peak_val = signal[new_bin]
                    boundary_val = (1 - ori_threshold) * peak_val
                    left = new_bin
                    while left > 0 and signal[left] >= boundary_val:
                        left -= 1
                    right = new_bin
                    while right < len(signal) and signal[right] >= boundary_val:
                        right += 1

                elif "TER" in name and minima.size > 0:
                    rel_idx = np.argmin(np.abs(minima - (center_bin - left_bin)))
                    new_bin = minima[rel_idx] + left_bin
                    dip_val = signal[new_bin]
                    boundary_val = (1 - ter_threshold) * dip_val
                    left = new_bin
                    while left > 0 and signal[left] <= boundary_val:
                        left -= 1
                    right = new_bin
                    while right < len(signal) and signal[right] <= boundary_val:
                        right += 1

                else:
                    new_bin = center_bin
                    left, right = new_bin, new_bin + 1

                new_center = new_bin * stride
                new_start = max(0, left * stride)
                new_end = min(chrom_size, right * stride)

                recentered.append((chrom, new_start, new_end, name, new_center))

    # Write BED6
    with open(out_file, "w") as f:
        for chrom, start, end, name, center in recentered:
            strand = "+" if "ORI" in name else "-"
            f.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t{strand}\n")

    return recentered

def quantify_ori_term_efficiency_from_bw(
    recentered_candidates: List[Tuple[str, int, int, str, int]],
    input_files: List[str],
    out_prefix: str,
    eval_resolution: int,
    out_file: Optional[str] = None
) -> List[Tuple[str, int, int, str, float, int]]:
    """
    Quantify ORI/TER efficiency from an OEM BigWig at a given resolution.

    Each candidate is scored as:
      - ORI: maximum OEM in candidate region
      - TER: minimum OEM in candidate region

    The BED score is scaled as round(1000 * abs(score)).

    Parameters
    ----------
    recentered_candidates : list of tuples
        [(chrom, start, end, name, center), ...] from recentering step.
    input_files : list of str
        OEM BigWig file paths, following pattern: id_oem_stride_resolution.bw
    out_prefix : str
        Prefix for BED output if out_file is None.
    eval_resolution : int
        Resolution (bp) used to select the correct OEM BigWig file.
    out_file : str, optional
        Output BED-like file.

    Returns
    -------
    efficiency_lines : list of tuples
        [(chrom, start, end, name, score, center), ...]
    """
    if out_file is None:
        out_file = f"{out_prefix}_efficiency_candidates.bed"

    # Select OEM file with matching resolution
    oem_files = [
        f for f in input_files
        if re.search(rf"_oem_\d+_{eval_resolution}\.bw$", os.path.basename(f), re.IGNORECASE)
    ]
    if not oem_files:
        raise ValueError(f"No OEM BigWig file found for resolution={eval_resolution} in input_files.")
    oem_file = oem_files[0]

    # Parse stride from filename
    match = re.search(r"_oem_(\d+)_(\d+)\.bw$", os.path.basename(oem_file), re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse stride/resolution from OEM file: {oem_file}")
    stride = int(match.group(1))
    resolution = int(match.group(2))
    assert resolution == eval_resolution, f"File resolution {resolution} ≠ eval_resolution {eval_resolution}"

    efficiency_lines: List[Tuple[str, int, int, str, float, int]] = []

    with pyBigWig.open(oem_file) as oem_bw:
        for chrom in oem_bw.chroms():
            chrom_size = oem_bw.chroms()[chrom]
            per_base_signal = np.nan_to_num(oem_bw.values(chrom, 0, chrom_size, numpy=True))
            signal = per_base_signal[::stride]

            # Candidates from this chromosome
            chrom_candidates = [c for c in recentered_candidates if c[0] == chrom]

            for _, start, end, name, center in chrom_candidates:
                left_bin = max(0, start // stride)
                right_bin = min(signal.size, (end + stride - 1) // stride)
                if right_bin <= left_bin:
                    continue

                local_signal = signal[left_bin:right_bin]

                if "ORI" in name:
                    score = float(np.max(local_signal))
                    if score < 0:
                        continue
                elif "TER" in name:
                    score = float(np.min(local_signal))
                    if score > 0:
                        continue
                else:
                    continue

                efficiency_lines.append((chrom, start, end, name, score, center))

    # Helper to assign BED RGB colors
    def colorize(name: str) -> str:
        return "0,255,0" if "ORI" in name else "255,0,0"

    # Write BED-like output
    with open(out_file, "w") as f:
        for chrom, start, end, name, score, center in efficiency_lines:
            strand = "+" if "ORI" in name else "-"
            bed_score = int(round(1000 * abs(score)))  # OEM in [0,1] → [0,1000]
            color = colorize(name)
            f.write(f"{chrom}\t{start}\t{end}\t{name}\t{bed_score}\t{strand}\t"
                    f"{start}\t{end}\t{color}\n")

    return efficiency_lines

def filter_efficiency_candidates(
	candidates: List[Tuple[str, int, int, str, float, int]],
	cutoff: float = 10,
	out_file: Optional[str] = None
) -> List[Tuple[str, int, int, str, float, int]]:
	"""
	Filter efficiency candidates by a BED score cutoff.

	Parameters
	----------
	candidates : list of tuples
		Each tuple: (chrom, start, end, name, score, center)
		Score is the OEM value in [-1,1].
	cutoff : float, optional
		Minimum BED score (0–1000) to retain. Default is 10.
	out_file : str, optional
		If provided, writes filtered candidates in BED6+ format.

	Returns
	-------
	filtered : list of tuples
		Filtered candidates: (chrom, start, end, name, score, center)
	"""
	filtered: List[Tuple[str, int, int, str, float, int]] = []

	for chrom, start, end, name, score, center in candidates:
		bed_score = int(round(1000 * abs(score)))
		if bed_score >= cutoff:
			filtered.append((chrom, start, end, name, score, center))

	if out_file:
		with open(out_file, "w") as f:
			for chrom, start, end, name, score, center in filtered:
				strand = "+" if "ORI" in name else "-"
				color = "0,255,0" if "ORI" in name else "255,0,0"
				bed_score = int(round(1000 * abs(score)))
				f.write(f"{chrom}\t{start}\t{end}\t{name}\t{bed_score}\t{strand}\t"
						f"{start}\t{end}\t{color}\n")

	return filtered