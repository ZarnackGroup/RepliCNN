*This reduced dataset is meant for testing of the replicnn functionality and whether the installation worked out.*

Create multi-resolution RFD and OEM tracks for chrI and chrII of GSM3939124:
```
for res in 5000 10000 15000; do
	for chr in chrI chrII; do
		for task in rfd oem; do
			replicnn rfd_oem \
						-w GSM3939124.${chr}.fwd.bw \
						-c GSM3939124.${chr}.rev.bw \
						-cs sacCer3.${chr}.chrom.sizes \
						-o GSM3939124.${chr} \
						-res ${res} \
						-st 1 \
						-t ${task} \
						-inv
		done
	done
done
```

Call ORIs and TERMs for chrI and chrII of GSM3939124:
```
for chr in chrI chrII; do
	replicnn ori_ter \
		--input GSM3939124.${chr}_oem_1_5000.bw GSM3939124.${chr}_oem_1_10000.bw GSM3939124.${chr}_oem_1_15000.bw GSM3939124.${chr}_rfd_1_5000.bw GSM3939124.${chr}_rfd_1_10000.bw GSM3939124.${chr}_rfd_1_15000.bw \
		--chromsizes sacCer3.${chr}.chrom.sizes \
		--output_prefix GSM3939124.${chr} \
		--save_intermediates \
		--ori-threshold 0.01 \
		--ter-threshold 0.1 \
		--window-radius 5000 \
		--max-merge-size 5000 \
		--n-evidence 2 \
		--eval_resolution 10000 \
		--cutoff 10
done
```

Prepare files for chrI and chrII of GSM3939124:
```
for chr in chrI chrII; do
	replicnn prepare \
		-fwd GSM3939124.${chr}.fwd.bw \
		-rev GSM3939124.${chr}.fwd.bw \
		-bs 500 \
		-cs sacCer3.${chr}.chrom.sizes \
		-o GSM3939124.${chr}.tsv \
		-t GSE36045_BY4743.${chr}.timing.bg \
		--invert
done
```

Train on RT of chrII of GSM3939124:
```
replicnn train \
    -i GSM3939124.chrII.tsv \
    -o GSM3939124_model \
    -g \
	-ws 201 \
	-e 300 \
	-bs 8192 \
	-v 0.1 \
	-lr 0.001
```

Predict RT of chrI of GSM3939124:
```
replicnn predict \
    -i GSM3939124.chrI.tsv \
    -m GSM3939124_model \
    -o GSM3939124.chrI.prediction.tsv \
    -g
```