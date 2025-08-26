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

# import packages
import sys
import typing
import logging

import argparse
from importlib.metadata import version, distribution
from .utils.logger import get_logger

# make sure there is a logger
logger: logging.Logger = get_logger(level=logging.DEBUG)

def main(logger:logging.Logger=get_logger(level=logging.DEBUG)) -> None:
	"""Main function that wraps all submodules as commandline callable commands."""

	# create parser
	parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="replicnn",
															  description="RepliCNN - Replication timing prediction and analyses",
															  )
	parser.add_argument("-v", "--version", action="version", version=f"RepliCNN v{version('replicnn')}")
	subparsers = parser.add_subparsers(help="Commands", dest="command")

	# create subparser for module 0: prepare
	## parser
	prepare_parser: argparse.ArgumentParser = subparsers.add_parser("prepare",
																	help="Prepare data format for this tool.", 
																	description="RepliCNN prepare - Prepare a file in the SDF format for usage in the tool and user specific analyses.",
																	)
	## arguments									 
	prepare_parser.add_argument("-fwd", "--forward", 
								help="Path to the forward bigWig file.", 
								type=str, nargs=1, required=True)
	prepare_parser.add_argument("-rev", "--reverse", 
								help="Path to the reverse bigWig file.", 
								type=str, nargs=1, required=True)
	prepare_parser.add_argument("-bs", "--binsize", 
								help="Binsize to use.", 
								type=int, nargs=1, required=True)
	prepare_parser.add_argument("-cs", "--chromsizes", 
								help="Path to a chromsizes file.", 
								type=str, nargs=1, required=True)
	prepare_parser.add_argument("-o", "--outpath", 
								help="File where the output should be written to.", 
							 	type=str, nargs=1, required=True)
	prepare_parser.add_argument("-t", "--timing", 
								help="Path to a timing file.", 
								type=str, nargs=1)
	prepare_parser.add_argument("-i", "--invert", 
								help="Invert phasing of the track.", 
								action="store_true")
	prepare_parser.add_argument("-nl", "--nolog", 
								help="Disable logging.", 
								action="store_false")

	# create subparser for module 1: train
	## parser
	train_parser: argparse.ArgumentParser = subparsers.add_parser("train", 
																  help="Train a model.", 
																  description="RepliCNN train - Train a model using SDF-file(s). Model quality can be assessed using the -cv option performing a Leave-One-Chromosome-Out Cross-Validation.",
																  )
	## arguments
	train_parser.add_argument("-i", "--input", 
							  help="Path(-s) to one/multiple sdf file(-s).", 
							  type=str, nargs="+", required=True)
	train_parser.add_argument("-o", "--outpath", 
							  help="Folder where the model should be written to.", 
							  type=str, nargs=1, required=True)
	train_parser.add_argument("-g", "--gpu", 
							  help="Enables training on gpu. Defaults to False", 
							  action="store_true")
	train_parser.add_argument("-ws", "--windowsize", 
							  help="Window size for chunks. Defaults to 201.", 
							  type=int, nargs=1)
	train_parser.add_argument("-e", "--epochs", 
							  help="Number of epochs to train for. Defaults to 300.", 
							  type=int, nargs=1)
	train_parser.add_argument("-bs", "--batchsize", 
							  help="Batch size. Defaults to 32.", 
							  type=int, nargs=1)
	train_parser.add_argument("-nes", "--noearlystopping", 
							  help="Whether to inactivate early stopping during training. Defaults to False.", 
							  action="store_false")					  
	train_parser.add_argument("-v", "--validationsplit", 
							  help="Percent of data used as validation. Defaults to 0.1.", 
							  type=float, nargs=1)
	train_parser.add_argument("-lr", "--learningrate", 
							  help="Learning rate for Adam optimizer. Defaults to 0.001.", 
							  type=float, nargs=1)
	train_parser.add_argument("-cv", "--crossvalidate", 
							  help="Leave-One-Chromosome-Out Cross-Validation on the given dataset. Only compatible with one SDF-file.", 
							  action="store_true")
	train_parser.add_argument("-nl", "--nolog", 
							  help="Disable logging.", 
							  action="store_false")		  	  

	# create subparser for module 2: predict
	## parser
	predict_parser: argparse.ArgumentParser = subparsers.add_parser("predict", 
																	help="Predict timing for file.", 
																	description="RepliCNN predict - Predict timing for a SDF-file using a previously trained model.",
																	)
	## arguments
	predict_parser.add_argument("-i", "--input", 
								help="Path to one sdf-file.", 
								type=str, nargs=1, required=True)
	predict_parser.add_argument("-m", "--modelpath", 
								help="Path to a model file.", 
								type=str, nargs=1, required=True)
	predict_parser.add_argument("-o", "--outpath", 
								help="File where the output should be written to.", 
								type=str, nargs=1)
	predict_parser.add_argument("-g", "--gpu", 
								help="Enables prediction on gpu. Defaults to False", 
							  	action="store_true")	
	predict_parser.add_argument("-nl", "--nolog", 
								help="Disable logging.", 
								action="store_false")	

	# create subparser for module 3: analyse
	## parser
	analyse_parser: argparse.ArgumentParser = subparsers.add_parser("analyse", 
																	help="Analyse data for characteristics.", 
																	description="RepliCNN analyse - Analyse a SDF-file for origins of rpelication/initiation zones, termination zones, constant timing regions, timing transition regions, and replication for directionality.",
																	)
	## arguments
	analyse_parser.add_argument("-i", "--input", 
								help="Path(-s) to one sdf file.", 
								type=str, nargs=1, required=True)
	analyse_parser.add_argument("-o", "--outpath", 
								help="Folder where the output should be written to.", 
								type=str, nargs=1, required=True)
	analyse_parser.add_argument("-s1", "--smoothlog2", 
								help="Smoothing factor for ORI and TERM identification.", 
								type=str, nargs=1)																
	analyse_parser.add_argument("-s2", "--smoothrfd", 
								help="Smoothing factor for TTR and RFD identification", 
								type=str, nargs=1)
	analyse_parser.add_argument("-nl", "--nolog", 
								help="Disable logging.", 
								action="store_false")	

	# create subparser for module 4: quantify
	## parser
	quantify_parser: argparse.ArgumentParser = subparsers.add_parser("quantify", 
																	 help="Quantify timing changes.", 
																	 description="RepliCNN quanitfy - Quantify timing changes between two conditions or one condition and a reference.",
																	 )
	## arguments
	quantify_parser.add_argument("-c1", "--cond1", 
								 help="List of SDF-files for conditions 1.", 
								 type=str, nargs="+", required=True)
	quantify_parser.add_argument("-c2", "--cond2", 
								 help="List of SDF-files for conditions 2. Mutually exclusive with reference.", 
								 type=str, nargs="+")
	quantify_parser.add_argument("-ref", "--reference", 
								 help="Reference timing file. Mutually exclusive with cond2.", 
								 type=str, nargs=1)
	quantify_parser.add_argument("-a", "--alpha", 
								 help="Alpha threshold for significance testing. Defaults to 0.05.", 
								 type=float, nargs=1)
	quantify_parser.add_argument("-bed", "--asbedfile", 
								 help="Write output in bed6-format.", 
								 action="store_true")
	quantify_parser.add_argument("-nl", "--nolog", 
								 help="Disable logging.", 
								 action="store_false")	

	# create subparser for module 5: visualise
	## parser
	# visualise_parser: argparse.ArgumentParser = subparsers.add_parser("visualise", 
	# 																  help="Visualise data.", 
	# 																  description="RepliCNN visualise - Visualise the dataformats and logs previously generated with the other submodules.",
	# 																  )
	## arguments
	# visualise_parser.add_argument("-o", "--outpath", 
	# 							  help="Folder where the output should be written to.", 
	# 							  type=str, nargs=1)

	# parse arguments and check if arguments are available, else print help
	args: argparse.Namespace = parser.parse_args()

	if len(sys.argv) < 2:
		parser.parse_args(["--help"])
		args.command == "error"
	
	# start chosen module and parse arguments
	## module 0: prepare
	if args.command == "prepare":
		try:
			logger.info("Started RepliCNN prepare!")
			from .prepare import _prepare
			_prepare(
				path_fwd = args.forward[0], 
				path_rev = args.reverse[0], 
				binsize = args.binsize[0], 
				path_chromsizes = args.chromsizes[0],
				path_out = args.outpath[0], 
				path_time = args.timing[0] if args.timing else "",
				invert = args.invert,
				log = args.nolog,
			)
		finally:
			logger.info("Ended RepliCNN prepare!")

	## module 1: train
	elif args.command == "train":
		try:
			logger.info("Started RepliCNN train!")
			from .train import _train
			_train(
				paths_sdfs = args.input, 
				path_out = args.outpath[0], 
				gpu = args.gpu, 
				windowsize = args.windowsize[0], 
				epochs = args.epochs[0], 
				batch_size = args.batchsize[0], 
				early_stopping = args.noearlystopping, 
				val_split = args.validationsplit[0], 
				learning_rate = args.learningrate[0], 
				cv = args.crossvalidate,
				log = args.nolog,
			)
		finally:
			logger.info("Ended RepliCNN train!")

	## module 2: predict
	elif args.command == "predict":
		try:
			logger.info("Started RepliCNN predict!")
			from .predict import _predict
			_predict(
				path_sdf = args.input[0],
				path_model = args.modelpath[0],
				path_out = args.outpath[0] if args.outpath else "",
				gpu = args.gpu,
				log = args.nolog,
			)
		finally:
			logger.info("Ended RepliCNN predict!")

	## module 3: analyse
	elif args.command == "analyse":
		try:
			logger.info("Started RepliCNN analyse!")
			from .analyse import _analyse
			_analyse(
				path_sdf = args.input[0],
				path_out = args.outpath[0],
				smoothing_factor_log2 = args.smoothlog2[0] if args.smoothlog2 else "",
				smoothing_factor_rfd = args.smoothrfd[0] if args.smoothrfd else "",
				log = args.nolog,
			)
		finally:
			logger.info("Ended RepliCNN analyse!")

	## module 4: quantify
	elif args.command == "quantify":
		try:
			logger.info("Started RepliCNN quantify!")
			from .quantify import _quantify
			_quantify(
				paths_sdf_c1 = args.cond1,
				paths_sdf_c2 = args.cond2,
				path_timing = args.reference, 
				alpha = args.alpha, 
				as_bed = args.asbedfile, 
				log = args.nolog,
			)
		finally:
			logger.info("Ended RepliCNN quantify!")

	# ## module 5: visualise
	# elif args.command == "visualise":
	# 	raise NotImplementedError("This function is not implemented yet. Please be patient.")
	# 	try:
	# 		logger.info("Started RepliCNN visualise!")
	# 		from .visualise import visualise
	# 		_visualise(
	# 			arg1 = args.arg1
	# 		)
	# 	finally:
	# 		logger.info("Ended RepliCNN visualise!")

	## something went wrong
	else:
		logger.error("No valid submodule chosen.")
		raise RuntimeError("Please choose a valid submodule.")
		
	return None

def main_wrapper():
    logger.info("Started RepliCNN!")
    try:
        main()
    finally:
        logger.info("Ended RepliCNN!")