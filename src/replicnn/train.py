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

os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras

from .utils.dataio import load_sdf, save_model, save_dataframe, save_train_history
from .utils.misc import get_free_gpu
from .utils.logger import get_logger
from .utils.ml_helper import sdfs_to_features_targets, get_model

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
def _train(**kwargs) -> None:
	"""Pass arguments to main function and handle output."""

	#load data
	sdfs: list[pd.DataFrame] = [load_sdf(path_sdf) for path_sdf in kwargs["paths_sdfs"]]

	# call main function
	output: tuple[keras.src.models.sequential.Sequential,keras.src.callbacks.history.History] = train(
		sdfs = sdfs, 
		gpu = kwargs["gpu"], 
		windowsize = kwargs["windowsize"], 
		epochs = kwargs["epochs"], 
		batch_size = kwargs["batch_size"], 
		early_stopping = kwargs["early_stopping"], 
		val_split = kwargs["val_split"], 
		learning_rate = kwargs["learning_rate"], 
		cv = kwargs["cv"],
		path_out = kwargs["path_out"],
		log = kwargs["log"], 
		)

	# handle output
	save_model(model=output[0], path=kwargs["path_out"])
	save_train_history(history=output[1], path=kwargs["path_out"])

	return None

# main function of this module
def train(
	sdfs:list[pd.DataFrame], 
	gpu:bool=False, 
	windowsize:int=201, 
	epochs:int=300, 
	batch_size:int=1024, 
	early_stopping:bool=True, 
	val_split:float=0.1, 
	learning_rate:float=0.001, 
	cv:bool=False,
	path_out:str="",
	log:bool=False, 
	logger:logging.Logger=get_logger(level=logging.DEBUG), 
	) -> tuple[keras.src.models.sequential.Sequential,keras.src.callbacks.history.History]:
	"""This module take timed sdf-files as input and trains a CNN model with it."""

	# check which devices are available and use the chosen/available one
	if gpu and torch.cuda.is_available():
		logger.info(f"GPU chosen, CUDA is available.")
		device = torch.device("cuda")
		# try to find a gpu with the least power consumption
		try:
			free_gpu = get_free_gpu()
			logger.info(f"Chose GPU with ID {free_gpu}.")
			torch.cuda.set_device(f"cuda:{free_gpu}")
		except:
			logger.warning(f"GPU selection failed, using standard device.")
		
	elif gpu and not torch.cuda.is_available():
		logger.warning(f"GPU chosen, CUDA is not available, fallback to CPU.")
		device = torch.device("cpu")
	else:
		logger.info(f"CPU chosen.")
		device = torch.device("cpu")

	# LOCO-CV routine
	if cv:
		logger.info(f"LOCO-CV routine chosen.")

		# raise exception if more than one sdf is supplied
		if len(sdfs)>1:
			raise NotImplementedError("LOCO-CV not implemented for more than one SDF-file!")

		# get list of chromosomes in sdf
		chromosomes: list[str] = sdfs[0].chromosome.unique()

		# copy sdf and set time column to empty
		sdf_prediction: pd.DataFrame = sdfs[0].copy().assign(time=np.nan)

		# list where to save histories
		histories: list[pd.DataFrame] = []

		# iterate through all chromosomes
		for chromosome in chromosomes:
			# split sdf in local training and test data
			sdf_train: pd.DataFrame = sdfs[0].query("chromosome!=@chromosome")
			sdf_test: pd.DataFrame = sdfs[0].query("chromosome==@chromosome")
			
			# get features and targets for training and test set
			X_train, y_train = sdfs_to_features_targets(sdfs=[sdf_train], windowsize=windowsize)
			X_test, y_test = sdfs_to_features_targets(sdfs=[sdf_test], windowsize=windowsize)

			# setup model
			model: keras.src.models.sequential.Sequential = get_model(shape_1=X_train.shape[1], shape_2=X_train.shape[2])
			optimizer: keras.src.backend.torch.optimizers.torch_adam.Adam = keras.optimizers.Adam(learning_rate=learning_rate)
			loss: keras.losses.MeanSquaredError = keras.losses.MeanSquaredError()
			model.compile(optimizer, loss=loss, metrics=["r2_score","root_mean_squared_error"])

			# set callbacks
			callbacks: list = []
			if early_stopping:
				callbacks.append(keras.callbacks.EarlyStopping(monitor="loss", 
															   min_delta=1e-3, 
															   patience=20,
															   verbose=0, 
															   mode="auto", 
															   restore_best_weights=True))

			# train model on local training set
			logger.info(f"Started model training for {chromosome}.")
			history = model.fit(X_train,
								y_train,
								epochs=int(epochs),
								validation_split=val_split,
								verbose=0,
								batch_size=batch_size,
								callbacks=callbacks)
			logger.info(f"Finished model training for {chromosome}.")

			# predict on local test set
			y_prediction: np.ndarray = model.predict(X_test).reshape(-1).round(3)

			# Log R2 score for chromosome
			logger.info(f"{chromosome}; R2={round(scipy.stats.pearsonr(y_prediction,y_test)[0]**2,3)}")

			# handle saving of prediction in copy of dataframe
			sdf_prediction.loc[sdf_test.index, "time"]: pd.DataFrame = pd.Series(y_prediction,index=sdf_test.index)

			# handle output
			save_train_history(history, f"{path_out}_{chromosome}")

		# handle output	
		sdf_prediction: pd.DataFrame = sdf_prediction.round(decimals=3)
		save_dataframe(sdf_prediction, f"{path_out}_cv.tsv")
		
	# convert data to features and targets
	X_train, y_train = sdfs_to_features_targets(sdfs=sdfs, windowsize=windowsize)

	# setup model
	model: keras.src.models.sequential.Sequential = get_model(shape_1=X_train.shape[1], shape_2=X_train.shape[2])
	optimizer: keras.src.backend.torch.optimizers.torch_adam.Adam = keras.optimizers.Adam(learning_rate=learning_rate)
	loss: keras.losses.MeanSquaredError = keras.losses.MeanSquaredError()
	model.compile(optimizer, loss=loss, metrics=["r2_score","root_mean_squared_error"])

	# set callbacks
	callbacks: list = []
	if early_stopping:
		callbacks.append(keras.callbacks.EarlyStopping(monitor="loss", 
													   min_delta=1e-3, 
													   patience=20,
													   verbose=0, 
													   mode="auto", 
													   restore_best_weights=True))

	# train model
	logger.info(f"Started model training.")
	history: keras.src.callbacks.history.History = model.fit(
		X_train,
		y_train,
		epochs=int(epochs),
		validation_split=val_split,
		verbose=0,
		batch_size=batch_size,
		callbacks=callbacks)
	logger.info(f"Finished model training.")

	return (model, history)