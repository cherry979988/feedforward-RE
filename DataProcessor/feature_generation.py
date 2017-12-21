__author__ = 'ZeqiuWu'
import sys
import os
import math
from multiprocessing import Process, Lock
from nlp_parse import parse
from ner_feature import pipeline, filter, pipeline_test

def get_number(filename):
    with open(filename) as f:
        count = 0
        for line in f:
            count += 1
        return count

def multi_process_parse(fin, fout, isTrain, nOfNones):
    file = open(fin, 'r')
    sentences = file.readlines()
    sentsPerProc = int(math.floor(len(sentences)*1.0/numOfProcesses))
    lock = Lock()
    processes = []
    out_file = open(fout, 'w', 0)
    for i in range(numOfProcesses):
        if i == numOfProcesses - 1:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:], out_file, lock, i, isTrain, nOfNones))
        else:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:(i+1)*sentsPerProc], out_file, lock, i, isTrain, nOfNones))
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join()
    out_file.close()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print 'Usage:feature_generation.py -DATA -numOfProcesses -emtypeFlag(0 or 1) -negWeight (1.0) -nOfNones (1)'
        exit(1)
    indir = 'data/source/%s' % sys.argv[1]
    if int(sys.argv[3]) == 1:
        outdir = 'data/intermediate/%s_emtype/rm' % sys.argv[1]
        requireEmType = True
    elif int(sys.argv[3]) == 0:
        outdir = 'data/intermediate/%s/rm' % sys.argv[1]
        requireEmType = False
    else:
        print 'Usage:feature_generation.py -DATA -numOfProcesses -emtypeFlag(0 or 1)'
        exit(1)
    outdir_em = 'data/intermediate/%s/em' % sys.argv[1]

    numOfProcesses = int(sys.argv[2])
    numOfNones = int(sys.argv[5])
    # NLP parse

    raw_train_json = indir + '/train.json'
    raw_test_json = indir + '/test.json'
    train_json = outdir + '/train_new.json'
    test_json = outdir + '/test_new.json'

    ### Generate features using Python wrapper (disabled if using run_nlp.sh)
    print 'Start nlp parsing'
    multi_process_parse(raw_train_json, train_json, True, numOfNones)

    print 'Train set parsing done'
    multi_process_parse(raw_test_json, test_json, False, 1)
    print 'Test set parsing done'

    print 'Start rm feature extraction'

    pipeline(train_json, indir + '/brown', outdir, requireEmType=requireEmType, isEntityMention=False)
    filter(outdir+'/feature.map', outdir+'/train_x.txt', outdir+'/feature.txt', outdir+'/train_x_new.txt')

    pipeline_test(test_json, indir + '/brown', outdir+'/feature.txt',outdir+'/type.txt', outdir, requireEmType=requireEmType, isEntityMention=False)
