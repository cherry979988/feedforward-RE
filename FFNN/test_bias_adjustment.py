import torch
import numpy as np
import scipy.io as sio
import random
import model.utils as utils
import model.pack as pack
import model.noCluster as noCluster
import sys
import os

if len(sys.argv) != 8:
    print('Usage: test_bias_adjustment.py -DATA -outputDropout(0.2) -inputDropout(0) -batchSize(20) -embLen(50) -ratio(0.1) -seed(1234)')
    exit(1)
# python3 test_bias_adjustment.py KBP 0.5 0.1 80 50 none 4

SEED = int(sys.argv[7])
print('Using Random Seed: '+str(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

dataset = sys.argv[1]
drop_prob = float(sys.argv[2])
repack_ratio = float(sys.argv[3])
bat_size = int(sys.argv[4])
embLen = int(sys.argv[5])
ratio = float(sys.argv[6])

train_file = './data/intermediate/' + dataset + '/rm/train.data'
dev_file = './data/intermediate/' + dataset + '/rm/dev.data'
test_file = './data/intermediate/' + dataset + '/rm/test.data'

feature_file = './data/intermediate/' + dataset + '/rm/feature.txt'
type_file = './data/intermediate/' + dataset + '/rm/type.txt'
type_file_test = './data/intermediate/' + dataset + '/rm/type_test.txt'
none_ind = utils.get_none_id(type_file)
print("None id:", none_ind)

label_distribution = utils.get_distribution(type_file)
label_distribution_test = utils.get_distribution(type_file_test)

word_size, pos_embedding_tensor = utils.initialize_embedding(feature_file, embLen)
_, type_size, _, _, _ = utils.load_corpus(train_file)

nocluster = noCluster.noCluster(embLen, word_size, type_size, drop_prob, label_distribution, label_distribution_test)
#nocluster.load_state_dict(torch.load('./dumped_models/KBP_bias_eq_log_distrib_orig_hyper/ffnn_dump_'+'_'.join(sys.argv[1:6])+'.pth'))

torch.cuda.set_device(0)
nocluster.cuda()
if_cuda = True

packer = pack.repack(repack_ratio, 20, if_cuda)

# bias adjustment
doc_size_test, _, feature_list_test, label_list_test, type_list_test = utils.load_corpus(test_file)

results = []
for randseed in range(1, 6):
    nocluster.load_state_dict(torch.load('./dumped_models/ffnn_dump_' + '_'.join(sys.argv[1:6]) +'_' + str(randseed) + '.pth'))
    for i in range(100):
        labels, feature = utils.shuffle_data(label_list_test, feature_list_test)
        eva_n = int(ratio * len(label_list_test))

        eva_label = labels[:eva_n]
        eva_label_distribution = utils.get_distribution_from_list(eva_label, max_index=25)

        test_label = labels[eva_n:]
        test_feature = feature[eva_n:]

        fl_t, of_t = packer.repack_eva(test_feature)

        nocluster.eval()
        scores = nocluster.test_with_bias(fl_t, of_t, eva_label_distribution)

        entropy = utils.calcEntropy(scores)
        ind = utils.calcInd(scores)

        f1score, recall, precision, val_f1, pn_precision, nprecision, nrecall, accuracy = utils.noCrossValidation(ind.data, entropy.data, test_label, ind.data, entropy.data, test_label, none_ind)
        #print('F1 = %.4f, recall = %.4f, precision = %.4f, pn_precision = %.4f, accuracy = %.4f' % (f1score, recall, precision, pn_precision, accuracy))

        results.append([f1score, recall, precision])

print('F1, recall, precision: ')
print(np.average(results, axis=0))
print(np.std(results, axis=0))
print(np.shape(results))