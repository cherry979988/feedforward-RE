data="KBP"
devices="3"

for i in 1 2 3 4 5
do
    echo "Run #$i"
    CUDA_VISIBLE_DEVICES=$devices python3 FFNN/run.py $data 0.2 0.1 80
done

echo "testing for $data finished!"
