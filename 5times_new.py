import subprocess

devices = 2
dataset = 'KBP'
best_output_dropout = 0.5
best_input_dropout = 0.1
embLen = 50
bsize = 80
info = 'KBP_bias_eq_log_distrib_orig_hyper'#'bias_eq_log_distrib'#'NYT_bias_eq_log_distrib_orig_hyper'

for i in [1,2,3,4,5]:
    #cmd1 = 'CUDA_VISIBLE_DEVICES=%s python3 FFNN/run.py %s %s %s %s %d %d %s'\
    #    % (devices, dataset, best_output_dropout, best_input_dropout, bsize, embLen, i, info)
    #print(cmd1)
    #subprocess.call(cmd1,shell=True)

    cmd2 = 'CUDA_VISIBLE_DEVICES=%s python3 FFNN/test_manual.py %s %s %s %s %d %d %s'\
        % (devices, dataset, best_output_dropout, best_input_dropout, bsize, embLen, i, info)
    print(cmd2)
    subprocess.call(cmd2,shell=True)