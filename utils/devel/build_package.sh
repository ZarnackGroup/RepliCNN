#!/bin/bash

REPLICNN_VERSION="0.1.0"

cd /home/dos02bi/koenig_data/projects/rt_prediction/replicnn
conda remove -n replicnn_build_env --all -y
conda create -n replicnn_build_env python=3.13 pip -y && \
conda activate replicnn_build_env  && \
pip install build && \
python3 -m build  && \
conda deactivate

conda remove -n replicnn_build_env --all -y
conda remove -n replicnn_run_env --all -y
conda create -n replicnn_run_env python=3.13 pip -y  && \
conda activate replicnn_run_env  && \
python3 -m pip install dist/replicnn-${REPLICNN_VERSION}-py3-none-any.whl  && \
replicnn --help && \
conda deactivate

#apptainer build replicnn_${REPLICNN_VERSION}.sif apptainer.def
#apptainer run replicnn_${REPLICNN_VERSION}.sif --help