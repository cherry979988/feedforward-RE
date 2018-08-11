data=$1
devices=$2
dropout=$3
repack=$4

for i in 1 2 3 4 5
do
    echo "Run #$i"
    CUDA_VISIBLE_DEVICES=$devices python3 FFNN/run_w_tune.py $data $dropout $repack 80 $i
done

echo "testing for $data finished!"
