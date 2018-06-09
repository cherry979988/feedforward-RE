#!/usr/bin/env bash

Data=$1
echo $Data

mkdir -pv data/intermediate/$Data/em
mkdir -pv data/intermediate/$Data/rm
mkdir -pv data/results/$Data/em
mkdir -pv data/results/$Data/rm

### Generate features
echo 'Generate Features'
python DataProcessor/feature_generation.py $Data 5 0 1.0 1

python3 DataProcessor/cotype_data_transform.py --input data/intermediate/$Data/rm --output data/intermediate/$Data/rm
