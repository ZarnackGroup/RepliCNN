# RepliCNN <a href="https://github.com/zarnackgroup/replicnn"><img src="assets/replicnn_logo.png" alt="RepliCNN logo" align="right" width="150"/></a>

[![PyPI](https://img.shields.io/pypi/v/replicnn)](https://pypi.org/project/replicnn/)
[![Citation](https://img.shields.io/badge/CITE-RepliCNN%20(2025)-blue)](https://github.com/zarnackgroup/replicnn/blob/main/CITATION.cff)
[![License](https://img.shields.io/badge/license-GPL_3.0-green)](https://github.com/zarnackgroup/replicnn/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

RepliCNN is a tool for predicting replication timing from GLOE-Seq, TrAEL-Seq, or OK-Seq data using convolutional neural networks.

## Installation
We recommend installing RepliCNN via the PyPI using pip:
```bash
pip install replicnn
```

### Other installation/running options
<details>
<summary>Installing from source</summary>

You can install RepliCNN from source via:
```bash
pip install 'replicnn @ git+https://github.com/zarnackgroup/replicnn.git@main'
```
or
```bash
pip install 'replicnn @ git+ssh://git@github.com/zarnackgroup/replicnn.git@main'
```
</details>

<details>
<summary>Running as container</summary>

You can also use RepliCNN as a Docker/Singularity/Apptainer container. We provide pre-built containers as well as Dockerfiles and Singularity/Apptainer definition files. Ensure that you have Docker/Singularity/Apptainer available in your PATH.
```bash
# Using Docker
user@dev:/tmp$ docker run docker://ghcr.io/zarnackgroup/replicnn:0.1.0 --version
0.1.0

# Using Singularity
user@dev:/tmp$ singularity run docker://ghcr.io/zarnackgroup/replicnn:0.1.0 --version
0.1.0

# Using Apptainer
user@dev:/tmp$ apptainer run docker://ghcr.io/zarnackgroup/replicnn:0.1.0 --version
0.1.0
```
</details>

## Commands and how to use them
The main way how to use RepliCNN is through its command line interface. 
### replicnn
```bash
user@dev:/tmp$ replicnn --help
usage: replicnn [-h] [-v] {prepare,train,predict,rfd_oem,ori_ter} ...

RepliCNN - Replication timing prediction and analyses

positional arguments:
  {prepare,train,predict,rfd_oem,ori_ter}
                        Commands
    prepare             Prepare data format for this tool.
    train               Train a model.
    predict             Predict timing for file.
    rfd_oem             Compute RFD or OEM tracks from Watson/Crick BigWig files.
    ori_ter             Detect replication origins (ORIs) and termination zones (TERMs) from RFD/OEM tracks.

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
```
For additional help and documentation, please check out `replicnn --help` or `replicnn {prepare,train,predict,rfd_oem,ori_ter} --help` or the corresponding publication.

### Subcommands
<details>
<summary>replicnn prepare</summary>

```bash
user@dev:/tmp$ replicnn prepare --help
usage: replicnn prepare [-h] -fwd FORWARD -rev REVERSE -bs BINSIZE -cs CHROMSIZES -o OUTPATH [-t TIMING] [-i] [-nl]

RepliCNN prepare - Prepare a file in the SDF format for usage in the tool and user specific analyses.

options:
  -h, --help            show this help message and exit
  -fwd, --forward FORWARD
                        Path to the forward bigWig file.
  -rev, --reverse REVERSE
                        Path to the reverse bigWig file.
  -bs, --binsize BINSIZE
                        Binsize to use.
  -cs, --chromsizes CHROMSIZES
                        Path to a chromsizes file.
  -o, --outpath OUTPATH
                        File where the output should be written to.
  -t, --timing TIMING   Path to a timing file.
  -i, --invert          Invert phasing of the track.
  -nl, --nolog          Disable logging.
```
</details>

<details>
<summary>replicnn train</summary>

```bash
user@dev:/tmp$ replicnn train --help
usage: replicnn train [-h] -i INPUT [INPUT ...] -o OUTPATH [-g] [-ws WINDOWSIZE] [-e EPOCHS] [-bs BATCHSIZE] [-nes] [-v VALIDATIONSPLIT] [-lr LEARNINGRATE] [-cv] [-nl]

RepliCNN train - Train a model using SDF-file(s). Model quality can be assessed using the -cv option performing a Leave-One-Chromosome-Out Cross-Validation.

options:
  -h, --help            show this help message and exit
  -i, --input INPUT [INPUT ...]
                        Path(-s) to one/multiple sdf file(-s).
  -o, --outpath OUTPATH
                        Folder where the model should be written to.
  -g, --gpu             Enables training on gpu. Defaults to False
  -ws, --windowsize WINDOWSIZE
                        Window size for chunks. Defaults to 201.
  -e, --epochs EPOCHS   Number of epochs to train for. Defaults to 300.
  -bs, --batchsize BATCHSIZE
                        Batch size. Defaults to 32.
  -nes, --noearlystopping
                        Whether to inactivate early stopping during training. Defaults to False.
  -v, --validationsplit VALIDATIONSPLIT
                        Percent of data used as validation. Defaults to 0.1.
  -lr, --learningrate LEARNINGRATE
                        Learning rate for Adam optimizer. Defaults to 0.001.
  -cv, --crossvalidate  Leave-One-Chromosome-Out Cross-Validation on the given dataset. Only compatible with one SDF-file.
  -nl, --nolog          Disable logging.
```
</details>

<details>
<summary>replicnn predict</summary>

```bash
user@dev:/tmp$ replicnn predict --help
usage: replicnn predict [-h] -i INPUT -m MODELPATH [-o OUTPATH] [-g] [-nl]

RepliCNN predict - Predict timing for a SDF-file using a previously trained model.

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Path to one sdf-file.
  -m, --modelpath MODELPATH
                        Path to a model file.
  -o, --outpath OUTPATH
                        File where the output should be written to.
  -g, --gpu             Enables prediction on gpu. Defaults to False
  -nl, --nolog          Disable logging.
```
</details>

<details>
<summary>replicnn oem_rfd</summary>

```bash
user@dev:/tmp$ replicnn rfd_oem --help
usage: replicnn rfd_oem [-h] -w WATSON -c CRICK -cs CHROMSIZES -o OUTPUT_PREFIX -res RESOLUTION -st STRIDE -t {rfd,oem} [-bg] [-nd] [-inv]

RepliCNN analyse - Compute replication fork directionality (RFD) or origin efficiency metric (OEM) from strand-specific BigWig files and write the results as BigWig or bedGraph.

options:
  -h, --help            show this help message and exit
  -w, --watson WATSON   Path to Watson strand BigWig file.
  -c, --crick CRICK     Path to Crick strand BigWig file.
  -cs, --chromsizes CHROMSIZES
                        Path to chromosome sizes file.
  -o, --output_prefix OUTPUT_PREFIX
                        Prefix for output file(s).
  -res, --resolution RESOLUTION
                        Window size in bp.
  -st, --stride STRIDE  Stride (step size in bp).
  -t, --track {rfd,oem}
                        Track to compute: 'rfd' or 'oem'.
  -bg, --bedgraph       Write output as bedGraph instead of BigWig.
  -nd, --no_norm_depth  Do not normalize depth balance.
  -inv, --invert        Swap Watson/Crick signals.
```
</details>

<details>
<summary>replicnn ori_ter</summary>

```bash
user@dev:/tmp$ replicnn ori_ter --help
usage: replicnn ori_ter [-h] -i INPUT [INPUT ...] -cs CHROMSIZES -o OUTPUT_PREFIX [-si] [-nl] [--ori-threshold ORI_THRESHOLD] [--ter-threshold TER_THRESHOLD] [--window-radius WINDOW_RADIUS] [--max-merge-size MAX_MERGE_SIZE] [--n-evidence N_EVIDENCE] [--smooth-factor-base SMOOTH_FACTOR_BASE] [--cutoff CUTOFF] -er EVAL_RESOLUTION

RepliCNN ori_ter - Detect ORI and TER zones, timing transition regions, and constant timing regions based on RFD/OEM tracks.

options:
  -h, --help            show this help message and exit
  -i, --input INPUT [INPUT ...]
                        Path(s) to RFD/OEM BigWig files.
  -cs, --chromsizes CHROMSIZES
                        Path to chromosome sizes file.
  -o, --output_prefix OUTPUT_PREFIX
                        Prefix for output file(s).
  -si, --save_intermediates
                        Save intermediate candidate and filtering files.
  -nl, --nolog          Disable debug logging.
  --ori-threshold ORI_THRESHOLD
                        Threshold for ORI recentering.
  --ter-threshold TER_THRESHOLD
                        Threshold for TER recentering.
  --window-radius WINDOW_RADIUS
                        Window radius (bp) for recentering around OEM extrema.
  --max-merge-size MAX_MERGE_SIZE
                        Maximum size (bp) for merging candidate regions.
  --n-evidence N_EVIDENCE
                        Minimum number of supporting evidences for a candidate.
  --smooth-factor-base SMOOTH_FACTOR_BASE
                        Smoothing factor for raw candidate generation.
  --cutoff CUTOFF       Cutoff for filtering efficiency scores.
  -er, --eval_resolution EVAL_RESOLUTION
                        OEM resolution used for recentering and scoring.
```
</details>

## Import RepliCNN into a python script/jupyter notebook
Besides the usage as a command line tool, RepliCNN can also be imported into a python script or jupyter notebook. The results of the commandline tool and the imported version are equivalent.

```bash
user@dev:/tmp$ python -c "import replicnn; print(replicnn.__version__)"
0.1.0
```

## Getting help
If you've found a bug, would like to suggest a new feature or you have any issues regarding RepliCNN installation, walkthrough, and output interpretation please open a new [issue](https://github.com/zarnackgroup/replicnn/issues).

## Funding
This works was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) via project ID 393547839 – SFB 1361, to K.Z., H.D.U., V.R. and M.C.C., via project ID 533767322 – EXC 3113/1, Cluster for Nucleic Acid Sciences and Technologies – NUCLEATE, to K.Z., and via project ID 529989072 – CA 198/20-1, to M.C.C. We gratefully acknowledge the IMB Genomics Core Facility and its NextSeq 2000 sequencer (funded by the DFG – INST 247/870-1 FUGG).

## Acknowledgements
We would like to express our gratitude to the Genomics and Bioinformatics Core Facilities of the IMB gGmbH (Mainz, Germany) for their assistance in sequencing and data processing. We thank Nicolas Delhomme, Maximilian Reuter, Mario Keller and all members of the Zarnack group for helpful discussions.

## Citing
If you use RepliCNN in your research, please cite this project like this:

RepliCNN: High-resolution inference of the DNA replication program from strand-specific 3′ DNA end sequencing
Dominik Stroh, Nicola Zilio, Maruthi K. Pabba, Vassilis Roukos, M. Cristina Cardoso, Helle D. Ulrich, Kathi Zarnack
bioRxiv 2026.03.12.710907; doi: https://doi.org/10.64898/2026.03.12.710907 

BibTex:
```bibtex
@article {Stroh2026.03.12.710907,
	author = {Stroh, Dominik and Zilio, Nicola and Pabba, Maruthi K. and Roukos, Vassilis and Cardoso, M. Cristina and Ulrich, Helle D. and Zarnack, Kathi},
	title = {RepliCNN: High-resolution inference of the DNA replication program from strand-specific 3' DNA end sequencing},
	elocation-id = {2026.03.12.710907},
	year = {2026},
	doi = {10.64898/2026.03.12.710907},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/03/14/2026.03.12.710907},
	journal = {bioRxiv}
}
```