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

from importlib.metadata import version, distribution
__version__ = version("replicnn")
__license__ = distribution("replicnn").metadata["License"]

# from .prepare import prepare
# from .train import train
# from .predict import predict
# from .analyse import analyse
# from .quantify import quantify
# from .utils import dataio, misc, ml_helper, stats
# from .visualise import