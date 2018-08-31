import subprocess

devices = 6
dataset = 'KBP'
best_output_dropout = 0.1
best_input_dropout = 0.0
embLen = 50
bsize = 80

for i in [1,2,3,4,5]:
    cmd1 = 'CUDA_VISIBLE_DEVICES=%s python3 FFNN/run.py %s %s %s %s %d %d'\
        % (devices, dataset, best_output_dropout, best_input_dropout, bsize, embLen, i)
    print(cmd1)
    subprocess.call(cmd1,shell=True)

    cmd2 = 'CUDA_VISIBLE_DEVICES=%s python3 FFNN/test_manual.py %s %s %s %d %d'\
        % (devices, dataset, best_output_dropout, best_input_dropout, bsize, embLen, i)
    print(cmd2)
    subprocess.call(cmd2,shell=True)
