# RepliCNN <a href="https://github.com/zarnackgroup/replicnn"><img src="assets/replicnn_logo.png" alt="RepliCNN logo" align="right" width="150"/></a>

[![Citation](https://img.shields.io/badge/CITE-RepliCNN%20(2025)-blue)](https://github.com/zarnackgroup/replicnn/blob/main/CITATION.cff)
[![License](https://img.shields.io/badge/license-GPL_3.0-green)](https://github.com/zarnackgroup/replicnn/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)

RepliCNN is a tool for predicting replication timing from GLOE-Seq, TrAEL-Seq, or OK-Seq data using convolutional neural networks.

## Abstract
During S phase, the genome is replicated in a tightly regulated spatiotemporal order described as DNA replication timing. Discontinuous lagging-strand synthesis produces Okazaki fragments whose strand-specific distribution reflects replication dynamics. Here, we present RepliCNN, a deep-learning framework based on one-dimensional convolutional neural networks to predict replication timing from Okazaki fragment distributions obtained from strand-specific 3′ DNA end sequencing methods such as GLOE-Seq, TrAEL-seq, or OK-Seq. RepliCNN also automatically annotates replication origins, termination zones, replication fork directionality, and origin efficiency genome-wide from a single dataset generated from unsynchronized proliferating cells. Benchmarking on public and in-house human and yeast datasets using leave-one-chromosome-out cross-validation demonstrates high predictive accuracy, enabling comprehensive analyses of replication dynamics from strand-specific DNA 3′ end sequencing data.

## How to install RepliCNN
We recommend installing RepliCNN using pip directly from this repository:
```bash
pip install 'replicnn @ git+https://github.com/zarnackgroup/replicnn.git@main'
```
or
```bash
pip install 'replicnn @ git+ssh://git@github.com/zarnackgroup/replicnn.git@main'
```
<summary>Running as container</summary>

You can also use RepliCNN as a Docker/Singularity/Apptainer container. We will provide pre-built containers as well as Dockerfiles and Singularity/Apptainer definition files with the first official release. Ensure that you have Docker/Singularity/Apptainer available in your PATH.
```bash
# Using Docker
user@dev:/tmp$ docker run docker://ghcr.io/zarnackgroup/replicnn:1.0.0 --version
0.1.0

# Using Singularity
user@dev:/tmp$ singularity run docker://ghcr.io/zarnackgroup/replicnn:1.0.0 --version
0.1.0

# Using Apptainer
user@dev:/tmp$ apptainer run docker://ghcr.io/zarnackgroup/replicnn:1.0.0 --version
0.1.0
```
</details>

## What is RepliCNN and how to use it
The main way how to use RepliCNN is through its command line interface, although RepliCNN can also be used when imported into a python script or jupyter notebook.

The main workflow to generate replication timing predcitions is replicnn prepare -> train -> predict
The main workflow to genereate Iz and TERM predictions is replicnn oem_rfd -> ori_ter

### replicnn
The main way to call RepliCNN after installation is through `replicnn --help`. This will take the user to the main interface from which more subcommands can be called.
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
    ori_ter             Detect replication origins (ORI) and termination zones (TER) from RFD/OEM tracks.

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
```
For additional help and documentation, please check out `replicnn --help` or `replicnn {prepare,train,predict,rfd_oem,ori_ter} --help` or the corresponding publication.

### Subcommands
Below you will find more detailled explanation of the subcommands, their arguments, how they function, and what they do.

<details>
<summary>replicnn prepare</summary>
The `replicnn prepare` command converts strand-specific 3′ end sequencing data into the tsv format used by RepliCNN for model training, prediction, and downstream analyses.

This command should be used for sequencing assays from which **replication fork directionality (RFD)** can be calculated, including TrAEL-seq, GLOE-seq, OK-seq, and related methods.

#### Input data

RepliCNN expects two strand-specific **bigWig** files:

- **Forward strand** (`--forward`)
- **Reverse strand** (`--reverse`)

These files can be generated from aligned BAM files using tools such as [`bamCoverage`](https://deeptools.readthedocs.io/en/develop/content/tools/bamCoverage.html) from deepTools or equivalent software.

We recommend generating **unbinned** bigWig files, as RepliCNN performs the binning internally.

#### Choosing the bin size

The `--binsize` parameter determines the prediction resolution of RepliCNN.

As a general guideline, choose a bin size that results in approximately **10,000–300,000 genomic bins** across the genome of interest.

Typical values are:

| Organism | Recommended bin size |
|----------|----------------------:|
| *S. cerevisiae* | 500 bp |
| *H. sapiens* | 10 kb |
| *M. musculus* | 10 kb |

Smaller bin sizes increase resolution but also increase computational requirements.

#### Chromosome sizes

The `--chromsizes` file specifies the chromosomes processed by RepliCNN and their lengths.

Chromosome size files can be downloaded from the UCSC Genome Browser:
https://hgdownload.cse.ucsc.edu/goldenpath/<assembly>/bigZips/<assembly>.chrom.sizes

Only chromosomes included in this file will be used by RepliCNN during preprocessing, training, and prediction.

This allows users to define which chromosomes should be included and exclude unwanted sequences such as mitochondrial DNA or unplaced contigs.

#### Output

The `--outpath` argument specifies the output tsv file generated by `replicnn prepare`.

The resulting tsv file contains the processed strand-specific signal, calculated replication fork directionality, and (if provided) replication timing information.

#### Replication timing input (optional)

The `--timing` argument accepts a replication timing file in **bedGraph** format.

This file represents the experimentally determined replication timing values used as the training target for RepliCNN.

The resolution of the timing file does **not** need to match the selected `--binsize`. If necessary, RepliCNN interpolates the timing values to match the requested prediction resolution.

This parameter is optional and is not required when preparing data only for prediction.

#### RFD orientation and phasing

Depending on the experimental protocol, the calculated replication fork directionality (RFD) profile may have an inverted orientation.

The `--invert` option can be used to reverse the polarity of the RFD track.

After preprocessing, the RFD profile should be oriented such that by convention:

- replication origins / initiation zones correspond to transitions from **negative to positive** RFD values
- termination zones correspond to transitions from **positive to negative** RFD values

Correct orientation is important for accurate prediction of replication timing and initiation zones.

#### Command-line reference
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

#### Example:
```bash
replicnn prepare \
    --forward sample_forward.bw \
    --reverse sample_reverse.bw \
    --binsize 10000 \
    --chromsizes hg38.chrom.sizes \
    --timing replication_timing.bedgraph \
    --outpath sample.tsv
```
</details>

<details>
<summary>replicnn train</summary>
The `replicnn train` command is used to train a RepliCNN neural network model for predicting replication timing.

The input consists of one or multiple **SDF files** generated using [`replicnn prepare`](#replicnn-prepare). During training, RepliCNN learns the relationship between replication fork directionality (RFD) profiles and experimentally measured replication timing.

#### Input data

The `--input` parameter accepts one or multiple SDF files.

Multiple SDF files can be used to train a model across different datasets or biological conditions. The input files should contain compatible genome assemblies and preprocessing parameters.

#### Output

The `--outpath` parameter specifies the directory where the trained Keras model and associated training outputs are stored.

#### GPU acceleration

The `--gpu` option enables GPU-based model training if a compatible GPU is available.

GPU availability is automatically checked and reported in the log output.

Training on a GPU substantially reduces training time and is highly recommended.

#### Model parameters

#### Window size

The `--windowsize` parameter defines the number of neighboring genomic bins used as input context for predicting the replication timing of a given bin.

The window size determines how much local genomic information is provided to the model.

The same window size must be used during both training and prediction.

The default value is: `201`

#### Epochs

The `--epochs` parameter defines the number of training iterations over the complete training dataset.

The default value is: `300`

In most cases, the default value should be retained.

#### Batch size

The `--batchsize` parameter specifies the number of training samples processed simultaneously during model optimization.

Larger batch sizes can improve training speed but require more GPU memory.

The default value is: `32`

#### Early stopping

By default, RepliCNN uses early stopping during training to prevent overfitting.

The `--noearlystopping` option disables this behaviour.

Disabling early stopping is generally **not recommended**, as it may lead to reduced model generalization performance.

#### Validation split

The `--validationsplit` parameter defines the fraction of the training data held out for validation during training.

The validation set is not used for parameter optimization and is used to monitor model performance and detect overfitting.

The default value is: `0.1` (10%)

#### Learning rate

The `--learningrate` parameter controls the learning rate of the neural network optimizer.

RepliCNN uses the Adam optimizer.

The default value is: `0.001`

#### Cross-validation

The `--crossvalidate` option performs **Leave-One-Chromosome-Out Cross-Validation (LOCO-CV)** as described in the RepliCNN publication.

During LOCO-CV, one chromosome is excluded from training and used as an independent test set. This procedure provides an unbiased estimate of model performance on unseen genomic regions.

LOCO-CV currently requires a single tsv input file.

#### Command-line reference
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

#### Example
```bash
replicnn train \
    --input sample.tsv \
    --outpath trained_model \
    --gpu \
    --windowsize 201 \
    --epochs 300
```
</details>

<details>
<summary>replicnn predict</summary>
The `replicnn predict` command is used to predict replication timing from an tsv file using a previously trained RepliCNN model.

The model must first be generated using [`replicnn train`](#replicnn-train). The input tsv file should be generated using [`replicnn prepare`](#replicnn-prepare) with the same preprocessing parameters used during model training.

#### Input data

The `--input` parameter specifies the SDF file for which replication timing should be predicted.

The input data must contain the same type of strand-specific replication fork directionality (RFD) information and preprocessing resolution used during model training.

#### Model

The `--modelpath` parameter specifies the path to the trained RepliCNN model.

The model file is generated by [`replicnn train`](#replicnn-train) and contains the learned neural network parameters required for prediction.

#### Output

The `--outpath` parameter specifies the file where the predicted replication timing profile should be written.

The output contains the predicted replication timing values at the resolution defined during the `prepare` step.

#### GPU acceleration

The `--gpu` option enables GPU-based prediction if a compatible GPU is available.

GPU acceleration is highly recommended, as it substantially reduces inference time, especially for large genomes or high-resolution predictions.

#### Command-line reference
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

#### Example
```bash
replicnn predict \
    --input sample.tsv \
    --modelpath trained_model.keras \
    --outpath sample_prediction.tsv \
    --gpu
```
</details>

<details>
<summary>replicnn oem_rfd</summary>

The `replicnn rfd_oem` command is a utility for calculating **replication fork directionality (RFD)** and **origin efficiency metric (OEM)** tracks from strand-specific sequencing data.

The command takes strand-specific signal tracks in bigWig format and generates genome-wide RFD or OEM profiles for downstream visualization and replication analysis.

#### Input data

RepliCNN expects two strand-specific **bigWig** files:

- **Watson strand** (`--watson`)
- **Crick strand** (`--crick`)

These files can be generated from aligned BAM files using tools such as [`bamCoverage`](https://deeptools.readthedocs.io/en/develop/content/tools/bamCoverage.html) from deepTools or equivalent software.

We recommend generating **unbinned** bigWig files, as `replicnn rfd_oem` performs the required aggregation internally.

#### Chromosome sizes

The `--chromsizes` file specifies the chromosomes and chromosome lengths used for track generation.

Chromosome size files can be obtained from the UCSC Genome Browser: [ "https://hgdownload.cse.ucsc.edu/goldenpath/XXX/bigZips/XXX.chrom.sizes".](https://hgdownload.cse.ucsc.edu/goldenpath/<assembly>/bigZips/<assembly>.chrom.sizes)

Only chromosomes included in this file will be processed.

#### Output

The `--output_prefix` parameter defines the prefix used for the generated output files.

Depending on the selected options, the output is written as either:

- BigWig files (default)
- bedGraph files (`--bedgraph`)

#### Track type

The `--track` parameter determines which track is calculated:

- `rfd`  
  Calculate replication fork directionality.

- `oem`  
  Calculate the origin efficiency metric.

The mathematical definitions of these metrics are described in the RepliCNN publication.

#### Resolution

The `--resolution` parameter defines the genomic window size used for calculating the selected track.

The optimal resolution depends on the organism and the desired level of detail.

Recommended values:

| Organism | Recommended resolutions |
|----------|------------------------:|
| *H. sapiens* / *M. musculus* | 50,000–150,000 bp |
| *S. cerevisiae* | 2,500–15,000 bp |

Smaller resolutions provide a more detailed view of the replication landscape but are more sensitive to sequencing noise.

Larger resolutions capture broader replication patterns with reduced local detail.

#### Stride

The `--stride` parameter defines the step size between consecutive calculated values.

A stride of `1` calculates the track at every nucleotide position. Larger strides reduce output file size and computation time.

Recommended values:

| Organism | Recommended stride |
|----------|-------------------:|
| *S. cerevisiae* | 1–100 bp |
| *H. sapiens* / *M. musculus* | 10–1000 bp |

The choice of stride represents a trade-off between spatial resolution and file size.

#### Strand normalization

By default, RepliCNN normalizes the Watson and Crick strand signal depth.

This assumes that both input tracks have comparable overall signal strength.

The `--no_norm_depth` option disables this normalization.

This option should only be used if the input strand-specific tracks are already depth-balanced.

#### RFD orientation

Depending on the experimental protocol, the calculated RFD profile may have an inverted orientation.

The `--invert` option swaps the Watson and Crick signals.

After processing, RFD tracks should be oriented such that:

- replication origins / initiation zones correspond to transitions from **negative to positive** RFD values
- termination zones correspond to transitions from **positive to negative** RFD values

Correct orientation is required for consistent interpretation of replication fork directionality.

#### Command-line reference
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

#### Example
```bash
replicnn rfd_oem \
    --watson sample_forward.bw \
    --crick sample_reverse.bw \
    --chromsizes hg38.chrom.sizes \
    --output_prefix sample_rfd \
    --resolution 100000 \
    --stride 100 \
    --track rfd
```
</details>

<details>
<summary>replicnn ori_ter</summary>

The `replicnn ori_ter` command identifies **replication origins (ORIs)**, **initiation zones (IZs)**, and **termination zones (TERMs)** from replication fork directionality (RFD) and origin efficiency metric (OEM) tracks.

The method integrates information from multiple RFD/OEM tracks generated at different resolutions. Using multiple resolutions enables detection of both fine-scale and broad replication patterns, improving the robustness of ORI and TERM identification.

#### Input data

The `--input` parameter accepts one or multiple RFD/OEM BigWig files.

Multiple resolutions should generally be provided. Lower-resolution tracks capture broad replication patterns, while higher-resolution tracks allow detection of more localized replication features.

The number and range of resolutions can be adjusted depending on the genome size and the desired level of detail.

Example: `5 kb` `10 kb` `50 kb` `100 kb`

#### Chromosome sizes

The `--chromsizes` file specifies the chromosomes and chromosome lengths used during analysis.

Chromosome size files can be obtained from the UCSC Genome Browser: `https://hgdownload.cse.ucsc.edu/goldenpath/<assembly>/bigZips/<assembly>.chrom.sizes`

Only chromosomes included in this file are analyzed.

#### Output

The `--output_prefix` parameter defines the prefix used for all generated output files.

The analysis consists of several sequential processing steps. By default, only the final output files are retained.

The `--save_intermediates` option can be used to save intermediate candidate and filtering files generated during ORI/TERM identification.

#### ORI and TERM detection parameters

##### ORI and TERM thresholds

The `--ori-threshold` and `--ter-threshold` parameters define the signal fraction used for recentering ORI and TERM candidates.

The thresholds describe the fraction of the maximal OEM/RFD signal used to determine the final position of a candidate.

Recommended values:

| Feature | Recommended threshold |
|---------|----------------------:|
| ORI | 0.05 |
| TERM | 0.15 |

For example, an ORI threshold of `0.05` shifts the ORI center to the position where the signal decreases to 5% of the maximum peak signal.

#### Window radius

The `--window-radius` parameter defines the genomic radius around a candidate ORI or TERM region that is searched for a local signal extremum during recentering.

Increasing this value allows larger positional adjustments but may merge nearby events.

#### Candidate merging

The `--max-merge-size` parameter defines the maximum distance between candidate ORI/TERM regions that allows them to be merged.

This prevents closely spaced candidate regions from being treated as independent events.

#### Evidence requirement

The `--n-evidence` parameter defines the minimum number of input tracks in which an ORI or TERM candidate must be detected to be retained.

Because each input track corresponds to a different resolution, this parameter controls how many resolutions must support a candidate.

Higher values increase confidence but may reduce the number of detected events.

#### Evaluation resolution and scoring

The `--eval_resolution` parameter specifies the OEM track resolution used for final ORI/TERM evaluation and scoring.

Each candidate receives an OEM-based score according to bed file specifications ranging from: `0`-`999`

Higher scores indicate stronger OEM support and therefore higher confidence in the detected replication event.

#### Candidate filtering

The `--cutoff` parameter filters candidates based on their OEM score.

Candidates with scores below this threshold are removed from the final output.

#### Smoothing

The `--smooth-factor-base` parameter controls smoothing during spline approximation used for initial candidate detection.

Increasing this value reduces sensitivity to small signal fluctuations and results in smoother candidate profiles.

This can be useful for noisy datasets but may remove weak replication features.

#### Command-line reference
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

#### Example
```bash
replicnn ori_ter \
    --input rfd_5000.bw oem_5000.bw rfd_10000.bw oem_10000.bw \
    --chromsizes sacCer3.chrom.sizes \
    --output_prefix sample_ori_ter \
    --eval_resolution 10000
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
This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) via project ID 393547839 – SFB 1361, to K.Z., H.D.U., V.R., S.S. and M.C.C., via project ID 533767322 – EXC 3113/1, NUCLEATE – Cluster for Nucleic Acid Sciences and Technologies, to K.Z., and via project ID 529989072 – CA 198/20-1, to M.C.C. We gratefully acknowledge the IMB Genomics Core Facility and the use of its NextSeq500 (funded by the Deutsche Forschungsgemeinschaft [DFG; German Research Foundation]-INST 247/870-1 Forschungsgroßgeräte [FUGG]) and its NextSeq2000.

## Acknowledgements
We would like to express our gratitude to the Genomics and Bioinformatics Core Facilities of the IMB gGmbH (Mainz, Germany) for their assistance in the experiment, sequencing and data processing, especially María Camila Fetiva Mora, Giriram Mohana, and Frank Rühle for supporting the HAP1 TrAEL-seq experiment and for preprocessing the HAP1 TrAEL-seq data. We thank Nicolas Delhomme, Maximilian Reuter, Mario Keller, and all members of the Zarnack group for helpful discussions.

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