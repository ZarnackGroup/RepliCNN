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
import logging
import random

import pandas as pd
import numpy as np
import scipy

os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras

from .utils.dataio import load_sdf, load_model, save_dataframe
from .utils.logger import get_logger
from .utils.ml_helper import sdfs_to_features_targets

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

# make sure there is a logger
logger: logging.Logger = get_logger(level=logging.DEBUG)

# handler for commandline interface
def _predict(**kwargs) -> None:
	"""Pass arguments to main function and handle output."""

	# load data
	sdf: pd.DataFrame = load_sdf(kwargs["path_sdf"])
	model: keras.src.models.sequential.Sequential = load_model(kwargs["path_model"])
	
	# call main function
	sdf: pd.DataFrame = predict(
		sdf = sdf, 
		model = model,
		gpu = kwargs["gpu"], 
		log = kwargs["log"], 
		)

	# handle output
	if kwargs["path_out"]:
		save_dataframe(dataframe=sdf, path=kwargs["path_out"])
	else:
		save_dataframe(dataframe=sdf, path=kwargs["path_sdf"])

	return None

# main function of this module
def predict(
	sdf:pd.DataFrame, 
	model:keras.src.models.sequential.Sequential,
	gpu:bool=False, 
	log:bool=False, 
	logger:logging.Logger=get_logger(level=logging.DEBUG)
	) -> pd.DataFrame:
	"""This module takes a sdf-file and predicts the timing for it."""
	
	# format sdf-file to model friendly format
	X, y = sdfs_to_features_targets([sdf])

	# check which devices are available and use the chosen/available one
	if gpu and torch.cuda.is_available():
		logger.info(f"GPU chosen, CUDA is available.")
		device = torch.device("cuda")
	elif gpu and not torch.cuda.is_available():
		logger.warning(f"GPU chosen, CUDA is not available, fallback to CPU.")
		device = torch.device("cpu")
	else:
		logger.info(f"CPU chosen.")
		device = torch.device("cpu")

	# predict timing
	y: np.ndarray = model.predict(X).reshape(-1).round(3)

	# set predicted timing into sdf
	sdf: pd.DataFrame = sdf.assign(time=y)

	return sdf