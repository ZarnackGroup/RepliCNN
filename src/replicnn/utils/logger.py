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

import numpy as np
import scipy

# set seeds for reproducibility
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)

def get_logger(level:int=logging.DEBUG) -> logging.Logger:
	"""Creates a logger for the tool."""

	logger = logging.getLogger(__name__)
	logger.setLevel(level)
	
	if not logger.handlers:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s')
		console_handler.setFormatter(formatter)
		logger.addHandler(console_handler)

	return logger