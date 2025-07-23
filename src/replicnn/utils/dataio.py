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

#import matplotlib.pyplot as plt
#import seaborn as sns

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

# load data
def load_chromsizes(path:str, log:bool=False) -> pd.DataFrame:
	"""Loads the chromsizes-file located at the given path."""
	
	if log: logger.info(f"Loading chromsizes from: {path}")
	names: list[str] = ["chromosome", "length"]
	dtype: dict[str,str] = {"chromosome":"string",
							"length":"int64"}
	chromsizes: pd.DataFrame = pd.read_csv(path, sep="\t", header=None, names=names, dtype=dtype, encoding="utf-8", lineterminator="\n", decimal=".")
	
	chromsizes: pd.DataFrame = chromsizes.assign(start=0)[["chromosome","start","length"]]

	return chromsizes

def load_bg(path:str, score:str="score", log:bool=False) -> pd.DataFrame:
	"""Loads the bg-file located at the given path."""
	
	if log: logger.info(f"Loading bedgraph from: {path}")
	names: list[str] = ["chromosome", "start", "end", score]
	dtype: dict[str,str] = {"chromosome":"string",
							"start":"int64",
							"end":"int64",
							score:"float64"}
	bg: pd.DataFrame = pd.read_csv(path, sep="\t", header=None, names=names, dtype=dtype, encoding="utf-8", lineterminator="\n", decimal=".")
	
	return bg

def load_bed6(path:str="", log:bool=False) -> pd.DataFrame:
	"""Loads the bed6-file located at the given path."""
	
	names: list[str] = ["chromosome", "start", "end", "name", "score", "strand"]
	dtype: dict[str,str] = {"chromosome":"string",
							"start":"int64",
							"end":"int64",
							"name":"string",
							"score":"string",
							"strand":"string"}
	
	if path:
		if log: logger.info(f"Loading bed6 from: {path}")
		
		bed6: pd.DataFrame = pd.read_csv(path, sep="\t", header=None, names=names, dtype=dtype, encoding="utf-8", lineterminator="\n", decimal=".")
		try:
			bed6: pd.DataFrame = bed6.assign(score=bed6.score.astype(float))
		except:
			pass
	else:
		bed6: pd.DataFrame = pd.DataFrame(columns=names)

	return bed6

def load_bwa(path:str, score:str="score", log:bool=False) -> pd.DataFrame:
	"""Loads the bigWigAverageOverBed-file located at the given path."""
	
	if log: logger.info(f"Loading bwa-file from: {path}")
	names: list[str] = ["name","size","covered",score,"mean0","mean"]
	dtype: dict[str,str] = {"name":"string",
							"start":"int64",
							"end":"int64",
							score:"float64"}
	bwa: pd.DataFrame = pd.read_csv(path, sep="\t", header=None, names=names, dtype=dtype, encoding="utf-8", lineterminator="\n", decimal=".")
	bwa[["chromosome", "start", "end"]]: pd.DataFrame = bwa["name"].str.extract(r'([^:]+):(\d+)-(\d+)', expand=True)
	bwa: pd.DataFrame = bwa[["chromosome", "start", "end", score]]
	bwa["chromosome"]: pd.DataFrame = bwa.chromosome.astype(str)
	bwa["start"]: pd.DataFrame = bwa.start.astype(int)
	bwa["end"]: pd.DataFrame = bwa.end.astype(int)
	bwa[score]: pd.DataFrame = bwa[score].astype(float)

	return bwa

def load_sdf(path:str, log:bool=False) -> pd.DataFrame:
	"""Loads the SDF-file located at the given path."""
	
	if log: logger.info(f"Loading SDF from: {path}")
	names: list[str] = ["chromosome", "start", "end", "pos", "neg", "log2", "spline", "derivative", "antiderivative", "time"]
	dtype: dict[str,str] = {"chromosome":"string", 
							"start":"int64", 
							"end":"int64", 
							"pos":"float64", 
							"neg":"float64", 
							"log2":"float64",
							"spline":"float64",
							"derivative":"float64", 
							"antiderivative":"float64", 
							"time":"float64"}
	sdf: pd.DataFrame = pd.read_csv(path, sep="\t", header=None, names=names, dtype=dtype, encoding="utf-8", lineterminator="\n", decimal=".")

	return sdf

def load_tff(path:str, log:bool=False) -> pd.DataFrame:
	"""Loads the TFF-file located at the given path."""
	
	if log: logger.info(f"Loading TFF from: {path}")
	names: list[str] = ["chromosome", "start", "end", "times_c1", "mean_c1", "std_c1", "times_c2", "mean_c2", "std_c2", "p_value", "p_adj", "significant"]
	dtype: dict[str,str] = {"chromosome":"string", 
							"start":"int64", 
							"end":"int64", 
							"times_c1":"string", 
							"mean_c1":"float64", 
							"std_c1":"float64", 
							"times_c2":"string", 
							"mean_c2":"float64", 
							"std_c2":"float64", 
							"p_value":"float64", 
							"p_adj":"float64",
							"significant":"bool"}
	tff: pd.DataFrame = pd.read_csv(path, sep="\t", header=None, names=names, dtype=dtype, encoding="utf-8", lineterminator="\n", decimal=".")

	return tff

def load_model(path:str, log:bool=False) -> keras.src.models.sequential.Sequential:
	"""Loads the model at the given path."""
	
	if log: logger.info(f"Loading model from: {path}")
	model:keras.src.models.sequential.Sequential = keras.models.load_model(path)

	return model

def load_train_history(path:str, wide:bool=True, log:bool=False) -> None:
	"""Loads the log-file from the train module at the given path."""
	
	if log: logger.info(f"Loading train history from: {path}")
	if wide:
		train_history: pd.DataFrame = pd.read_csv(path, sep="\t", encoding="utf-8", lineterminator="\n", decimal=".")
	else:
		train_history: pd.DataFrame = pd.read_csv(path, sep="\t", encoding="utf-8", lineterminator="\n", decimal=".").melt(id_vars=["epoch"])

	return train_history

def load_analyse_log(path:str, log:bool=False) -> None:
	"""Loads the log-file from the analyse module at the given path."""
	
	if log: logger.info(f"Loading analyse log from: {path}")
	# TODO

	return log_analyse

# save data
def save_dataframe(dataframe:pd.DataFrame, path:str, log:bool=False) -> None:
	"""Saves the dataframe to the given path."""
	
	if log: logger.info(f"Saving dataframe to: {path}")
	dirname: str = os.path.dirname(path)
	dirname: str = dirname if dirname else "./"
	os.makedirs(dirname,exist_ok=True)
	dataframe.to_csv(path, sep="\t", header=None, index=False, encoding="utf-8", lineterminator="\n", decimal=".")
	
	return None

def save_train_history(history:keras.src.callbacks.history.History, path:str, log:bool=False) -> None:
	"""Saves the log-file from the train module to the given path."""
	
	if log: logger.info(f"Saving train history to: {path}.log")
	os.makedirs(os.path.dirname(path),exist_ok=True)
	history_df: pd.DataFrame = pd.DataFrame(history.history).assign(epoch=pd.Series(history.epoch)+1)
	history_df.to_csv(f"{path}.log", sep="\t", index=False, encoding="utf-8", lineterminator="\n", decimal=".")
	
	return None

def save_analyse_log(analyse_log:None, path:str, log:bool=False) -> None:
	"""Saves the log-file from the analyse to the given path."""
	
	if log: logger.info(f"Saving analyse log to: {path}")
	os.makedirs(os.path.dirname(path),exist_ok=True)
	
	return None

def save_model(model:keras.src.models.sequential.Sequential, path:str, log:bool=False) -> None:
	"""Saves the model to the given path."""
	
	if log: logger.info(f"Saving model to: {path}.keras")
	os.makedirs(os.path.dirname(path),exist_ok=True)
	if not path.endswith("/"): path += "/"
	model.save(f"{path}.keras")
	
	return None

# def save_plot(path:str, log:bool=False) -> None:
# 	"""Saves the plot to the given path."""
	
# 	if log: logger.info(f"Saving plot to: {path}")
# 	os.makedirs(os.path.dirname(path),exist_ok=True)
# 	plt.savefig(path)
	
# 	return None