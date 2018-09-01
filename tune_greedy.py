import sys
import os
import pickle
import subprocess

def get_best_output_dropout(dataset, input_dropout, bsize, embLen):
    fin = open('tune_log.pkl', 'rb')
    d = pickle.load(fin)
    best_val_f1 = 0
    best_output_dropout = 0
    for key in d:
        if key[0]==dataset and key[2]==input_dropout and key[3]==bsize and key[4]==embLen:
            if d[key]>best_val_f1:
                best_val_f1 = d[key]
                best_output_dropout = key[1]
    return best_output_dropout

def get_best_input_dropout(dataset, output_dropout, bsize, embLen):
    fin = open('tune_log.pkl', 'rb')
    d = pickle.load(fin)
    best_val_f1 = 0
    best_input_dropout = 0
    for key in d:
        if key[0]==dataset and key[1]==output_dropout and key[3]==bsize and key[4]==embLen:
            if d[key]>best_val_f1:
                best_val_f1 = d[key]
                best_input_dropout = key[2]
    return best_input_dropout

def get_best_bsize(dataset, output_dropout, input_dropout, embLen):
    fin = open('tune_log.pkl', 'rb')
    d = pickle.load(fin)
    best_val_f1 = 0
    best_bsize = 0
    for key in d:
        if key[0]==dataset and key[1]==output_dropout and key[2]==input_dropout and key[4]==embLen:
            if d[key]>best_val_f1:
                best_val_f1 = d[key]
                best_bsize = key[3]
    return best_bsize

output_dropout_list = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
input_dropout_list = [0.5, 0.4, 0.3, 0.2, 0.1]
bsize_list = [160,40,20]

dataset = sys.argv[1]
devices = sys.argv[2]
embLen = int(sys.argv[3])

default_input_dropout = 0.0
default_bsize = 80

tune_time_seed = 1234

for output_dropout in output_dropout_list:
    cmd1 = 'CUDA_VISIBLE_DEVICES=%s python3 FFNN/run.py %s %s %s %s %d %d'\
        % (devices, dataset, output_dropout, default_input_dropout, default_bsize, embLen, tune_time_seed)
    print(cmd1)
    subprocess.call(cmd1,shell=True)

best_output_dropout = get_best_output_dropout(dataset, default_input_dropout, default_bsize, embLen)

for input_dropout in input_dropout_list:
    cmd1 = 'CUDA_VISIBLE_DEVICES=%s python3 FFNN/run.py %s %s %s %s %d %d'\
        % (devices, dataset, best_output_dropout, input_dropout, default_bsize, embLen, tune_time_seed)
    print(cmd1)
    subprocess.call(cmd1,shell=True)

best_input_dropout = get_best_input_dropout(dataset, best_output_dropout, default_bsize, embLen)

# for bsize in bsize_list:
#     cmd1 = 'CUDA_VISIBLE_DEVICES=%s python3 FFNN/run.py %s %s %s %s %d %d'\
#         % (devices, dataset, best_output_dropout, best_input_dropout, bsize, embLen, tune_time_seed)
#     print(cmd1)
#     subprocess.call(cmd1,shell=True)    \

# best_bsize = get_best_bsize(dataset, best_output_dropout, best_input_dropout)
best_bsize = 80

print('====TUNING COMPLETED!====')
print('Best Param: Input Dropout = %s, Output Dropout = %s, Batch Size = %s' % (str(best_output_dropout), str(best_input_dropout), str(best_bsize)))

for i in [1,2,3,4,5]:
    cmd1 = 'CUDA_VISIBLE_DEVICES=%s python3 FFNN/run.py %s %s %s %s %d %d'\
        % (devices, dataset, best_output_dropout, best_input_dropout, best_bsize, embLen, i)
    print(cmd1)
    subprocess.call(cmd1,shell=True)
    
    cmd2 = 'CUDA_VISIBLE_DEVICES=%s python3 FFNN/test_manual.py %s %s %s %s %d %d'\
        % (devices, dataset, best_output_dropout, best_input_dropout, best_bsize, embLen, i)
    print(cmd2)
    subprocess.call(cmd2,shell=True)