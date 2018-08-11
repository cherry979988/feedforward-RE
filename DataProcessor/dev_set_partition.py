__author__ = 'QinyuanYe'
import sys
import random
from shutil import copyfile


# split the original train set into
# 90% train-set (train_split.json) and 10% dev-set (dev.json)
if __name__ == "__main__":
    random.seed(1234)
    
    if len(sys.argv) != 3:
        print 'Usage:feature_generation.py -DATA -ratio'
        exit(1)

    dataset = sys.argv[1]
    ratio = float(sys.argv[2])

    dir = 'data/source/%s' % sys.argv[1]
    original_train_json = dir + '/train.json'
    train_json = dir + '/train_split.json'
    dev_json = dir + '/dev.json'

    if dataset == 'TACRED' or dataset == 'NYT10':
        print '%s has a provided dev set, skip splitting' % dataset
        copyfile(original_train_json, train_json)
        exit(0)

    fin = open(original_train_json, 'r')
    lines = fin.readlines()
    dev_size = int(ratio * len(lines))

    random.shuffle(lines)

    dev = lines[:dev_size]
    train_split = lines[dev_size:]

    fout1 = open(dev_json, 'w')
    fout1.writelines(dev)

    fout2 = open(train_json, 'w')
    fout2.writelines(train_split)