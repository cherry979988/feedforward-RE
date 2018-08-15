data=$1
devices=$2
dropout=$3
repack=$4
runtype=$5

for i in 1 2 3 4 5
do
    echo "Run #$i"
    CUDA_VISIBLE_DEVICES=$devices python3 FFNN/run.py $data $dropout $repack 80 $i
    CUDA_VISIBLE_DEVICES=$devices python3 FFNN/run_tune_bias.py $data $dropout $repack 80 $i
done

echo "testing for $data finished!"
