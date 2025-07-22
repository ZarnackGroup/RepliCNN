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

from .logger import get_logger

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

def sdfs_to_features_targets(sdfs:list[pd.DataFrame], windowsize:int=201, test_chroms:list[str]=[], log:bool=False) -> np.ndarray[float]:
	"""Takes a list of sdf-files as pd.DataFrame and returns the features and targets in the right format for training/prediction."""
	# assign unique chromosome names to each samples chromosomes
	sdfs: list[pd.DataFrame] = [sdf.assign(chromosome=pd.Series(f"{i}_" + sdf["chromosome"].astype(str))) for i, sdf in enumerate(sdfs)]

	# combine all dataframes into a single dataframe
	sdf: pd.DataFrame = pd.concat(sdfs, axis=0).reset_index(drop=True)

	# reshape data into the format we need for the model
	sdf_chromosomes: dict = {}
	X_chromosomes: dict = {}
	y_chromosomes: dict = {}

	# iterate through all chromosomes
	for chromosome in sdf.chromosome.unique():
		# get data for current chromosome
		current_data: pd.DataFrame = sdf[sdf.chromosome.values==chromosome].reset_index(drop=True)

		# extract data as numpy arrays
		feature_1: np.ndarray = current_data["log2"].to_numpy()
		feature_2: np.ndarray = current_data["derivative"].to_numpy()
		feature_3: np.ndarray = current_data["antiderivative"].to_numpy()
		target_1: np.ndarray = current_data["time"].to_numpy()

		# reshape data
		feature_1: np.ndarray = feature_1.reshape((len(feature_1), 1))
		feature_2: np.ndarray = feature_2.reshape((len(feature_2), 1))
		feature_3: np.ndarray = feature_3.reshape((len(feature_3), 1))
		target_1: np.ndarray = target_1.reshape((len(target_1), 1))

		# stack data
		sdf_chromosomes[chromosome] = np.hstack((feature_1, feature_2, feature_3, target_1))

		# initialize lists
		X_chromosomes[chromosome]: list = list()
		y_chromosomes[chromosome]: list = list()

		# number of bins to the left and to the right of the bin to predict
		flank = int((windowsize-1)/2)

		# bins per chromosome
		sequences = sdf_chromosomes[chromosome]
		num_sequences = len(sequences)
		
		# iterate through each bin
		for i in range(0,num_sequences):
			# get start and end of chunk for the bin
			start = max(i-flank,0)
			end = min(i+flank+1,num_sequences)

			# extract each chunk and timing for the bin
			seq_x = sequences[start:end,:-1]
			seq_y = sequences[i,-1]

			# pad on the left if necessary
			if i < flank:
				pad_left = np.zeros((flank - i, sequences.shape[1] - 1))
				seq_x = np.vstack((pad_left, seq_x))

			# pad on the right if necessary
			if i + flank >= num_sequences:
				pad_right = np.zeros(((i + flank + 1) - num_sequences, sequences.shape[1] - 1))
				seq_x = np.vstack((seq_x, pad_right))

			# add chunk to the list of chunks of this chromosome
			X_chromosomes[chromosome].append(seq_x)
			y_chromosomes[chromosome].append(seq_y)
	
	# combine chromosomes
	X: np.ndarray = np.array([item for key, sublist in X_chromosomes.items() for item in sublist])
	y: np.ndarray = np.array([item for key, sublist in y_chromosomes.items() for item in sublist])

	return (X, y)

def get_model(shape_1:int, shape_2:int, seed:int=42, log:bool=False) -> keras.src.models.sequential.Sequential:
	"""Return a model with the correct architecture that is ready for training."""

	# set up model with input shape
	model: keras.src.models.sequential.Sequential = keras.Sequential()
	model.add(keras.Input(shape=(shape_1, shape_2)))

	# add noise layer during training
	model.add(keras.layers.GaussianNoise(0.1, seed=seed))

	# add first convolution-norm-dropout block
	model.add(keras.layers.Conv1D(filters=64, kernel_size=32, strides=1, padding="valid", activation=keras.activations.relu))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.5))

	# add second convolution-norm-dropout block
	model.add(keras.layers.Conv1D(filters=32, kernel_size=16, strides=1, padding="valid", activation=keras.activations.relu))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.5))

	# add third convolution-norm-dropout block
	model.add(keras.layers.Conv1D(filters=16, kernel_size=8, strides=1, padding="valid", activation=keras.activations.relu))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.5))

	# add flattening and dense neural network
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(50, activation=keras.activations.relu))
	model.add(keras.layers.Dense(20, activation=keras.activations.relu))
	model.add(keras.layers.Dense(1, activation=keras.activations.tanh))

	return model