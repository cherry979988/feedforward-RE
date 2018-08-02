data="KBP"
devices="3"

for bsize in 80 40 20 10
do
    echo "batch_size = $bsize"
    CUDA_VISIBLE_DEVICES=$devices python3 FFNN/run.py $data 0.2 0.1 $bsize
done

echo "tuning for $data finished!"
