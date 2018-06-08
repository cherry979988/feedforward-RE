#!/usr/bin/env bash

Data=$1
echo $Data

### Generate brown file (clusters raw.txt into 300 clusters)
cd data/source/$Data
python3 generateBClusterInput.py
cd ../../..
cd DataProcessor/brown-cluster/
make
./wcluster --text ../../data/source/$Data/bc_input.txt --c 300 --output_dir ../../data/source/$Data/brown-out
cd ../../
mv data/source/$Data/brown-out/paths data/source/$Data/brown