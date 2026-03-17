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
	rfd_oem_parser: argparse.ArgumentParser = subparsers.add_parser(
		"rfd_oem",
		help="Compute RFD or OEM tracks from Watson/Crick BigWig files.",
		description=(
			"RepliCNN analyse - Compute replication fork directionality (RFD) or origin efficiency metric (OEM) "
			"from strand-specific BigWig files and write the results as BigWig or bedGraph."
		),
	)
	## arguments
	rfd_oem_parser.add_argument(
		"-w", "--watson",
		help="Path to Watson strand BigWig file.",
		type=str,
		required=True
	)
	rfd_oem_parser.add_argument(
		"-c", "--crick",
		help="Path to Crick strand BigWig file.",
		type=str,
		required=True
	)
	rfd_oem_parser.add_argument(
		"-cs", "--chromsizes",
		help="Path to chromosome sizes file.",
		type=str,
		required=True
	)
	rfd_oem_parser.add_argument(
		"-o", "--output_prefix",
		help="Prefix for output file(s).",
		type=str,
		required=True
	)
	rfd_oem_parser.add_argument(
		"-res", "--resolution",
		help="Window size in bp.",
		type=int,
		required=True
	)
	rfd_oem_parser.add_argument(
		"-st", "--stride",
		help="Stride (step size in bp).",
		type=int,
		required=True
	)
	rfd_oem_parser.add_argument(
		"-t", "--track",
		help="Track to compute: 'rfd' or 'oem'.",
		type=str,
		choices=["rfd", "oem"],
		required=True
	)
	rfd_oem_parser.add_argument(
		"-bg", "--bedgraph",
		help="Write output as bedGraph instead of BigWig.",
		action="store_true"
	)
	rfd_oem_parser.add_argument(
		"-nd", "--no_norm_depth",
		help="Do not normalize depth balance.",
		action="store_false"
	)
	rfd_oem_parser.add_argument(
		"-inv", "--invert",
		help="Swap Watson/Crick signals.",
		action="store_true"
	)

	# create subparser for module 3: ori_ter
	## parser
	ori_ter_parser: argparse.ArgumentParser = subparsers.add_parser(
		"ori_ter",
		help="Detect replication origins (ORI) and termination zones (TER) from RFD/OEM tracks.",
		description=(
			"RepliCNN ori_ter - Detect ORI and TER zones, timing transition regions, and "
			"constant timing regions based on RFD/OEM tracks."
		),
	)

	## arguments
	ori_ter_parser.add_argument(
	"-i", "--input",
	help="Path(s) to RFD/OEM BigWig files.",
	type=str,
	nargs="+",
	required=True
	)
	ori_ter_parser.add_argument(
		"-cs", "--chromsizes",
		help="Path to chromosome sizes file.",
		type=str,
		required=True
	)
	ori_ter_parser.add_argument(
		"-o", "--output_prefix",
		help="Prefix for output file(s).",
		type=str,
		required=True
	)
	ori_ter_parser.add_argument(
		"-si", "--save_intermediates",
		help="Save intermediate candidate and filtering files.",
		action="store_true"
	)
	ori_ter_parser.add_argument(
		"-nl", "--nolog",
		help="Disable debug logging.",
		action="store_false"
	)
	ori_ter_parser.add_argument(
		"--ori-threshold",
		help="Threshold for ORI recentering.",
		type=float,
		default=0.05
	)
	ori_ter_parser.add_argument(
		"--ter-threshold",
		help="Threshold for TER recentering.",
		type=float,
		default=0.15
	)
	ori_ter_parser.add_argument(
		"--window-radius",
		help="Window radius (bp) for recentering around OEM extrema.",
		type=int,
		default=15000
	)
	ori_ter_parser.add_argument(
		"--max-merge-size",
		help="Maximum size (bp) for merging candidate regions.",
		type=int,
		default=15000
	)
	ori_ter_parser.add_argument(
		"--n-evidence",
		help="Minimum number of supporting evidences for a candidate.",
		type=int,
		default=2
	)
	ori_ter_parser.add_argument(
		"--smooth-factor-base",
		help="Smoothing factor for raw candidate generation.",
		type=float,
		default=1e-3
	)
	ori_ter_parser.add_argument(
		"--cutoff",
		help="Cutoff for filtering efficiency scores.",
		type=int,
		default=15
	)
	ori_ter_parser.add_argument(
		"-er", "--eval_resolution",
		help="OEM resolution used for recentering and scoring.",
		type=int,
		required=True
	)

	# create subparser for module 4: quantify
	# ## parser
	# quantify_parser: argparse.ArgumentParser = subparsers.add_parser("quantify", 
	# 																help="Quantify timing changes.", 
	# 																description="RepliCNN quanitfy - Quantify timing changes between two conditions or one condition and a reference.",
	# 																)
	# ## arguments
	# quantify_parser.add_argument("-c1", "--cond1", 
	# 							help="List of SDF-files for conditions 1.", 
	# 							type=str, nargs="+", required=True)
	# quantify_parser.add_argument("-c2", "--cond2", 
	# 							help="List of SDF-files for conditions 2. Mutually exclusive with reference.", 
	# 							type=str, nargs="+")
	# quantify_parser.add_argument("-ref", "--reference", 
	# 							help="Reference timing file. Mutually exclusive with cond2.", 
	# 							type=str, nargs=1)
	# quantify_parser.add_argument("-a", "--alpha", 
	# 							help="Alpha threshold for significance testing. Defaults to 0.05.", 
	# 							type=float, nargs=1)
	# quantify_parser.add_argument("-bed", "--asbedfile", 
	# 							help="Write output in bed6-format.", 
	# 							action="store_true")
	# quantify_parser.add_argument("-nl", "--nolog", 
	# 							help="Disable logging.", 
	# 							action="store_false")	

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

	## module 3: rfd_oem
	elif args.command == "rfd_oem":
		try:
			logger.info("Started RepliCNN rfd_oem!")
			from .rfd_oem import _rfd_oem  # import your _rfd_oem function
			_rfd_oem(
				watson_bw=args.watson,
				crick_bw=args.crick,
				chrom_sizes_file=args.chromsizes,
				resolution=args.resolution,
				stride=args.stride,
				output_prefix=args.output_prefix,
				track=args.track,
				bedgraph=args.bedgraph,
				norm_depth=args.no_norm_depth,
				invert=args.invert,
			)
		finally:
			logger.info("Ended RepliCNN rfd_oem!")

	## module 4: ori_ter
	elif args.command == "ori_ter":
		try:
			logger.info("Started RepliCNN ori_ter!")
			from .ori_ter import _ori_ter  # import your _ori_ter function
			_ori_ter(
				input_files=args.input,
				output_prefix=args.output_prefix,
				chrom_sizes_file=args.chromsizes,
				save_intermediates=args.save_intermediates,
				log=args.nolog,
				ori_threshold=args.ori_threshold,
				ter_threshold=args.ter_threshold,
				window_radius=args.window_radius,
				max_merge_size=args.max_merge_size,
				n_evidence=args.n_evidence,
				smooth_factor_base=args.smooth_factor_base,
				cutoff=args.cutoff,
				eval_resolution=args.eval_resolution,
			)
		finally:
			logger.info("Ended RepliCNN ori_ter!")

	## module 5: quantify
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

	# ## module 6: visualise
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