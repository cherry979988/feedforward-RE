import torch
import numpy as np
import random
import model.utils as utils
import model.pack as pack
import model.noCluster as noCluster
import sys

if len(sys.argv) != 6:
    print('Usage: run.py -DATA -outputDropout(0.2) -inputDropout(0) -batchSize(20) -randomseed(1234)')
    exit(1)

SEED = int(sys.argv[6])
print('Using Random Seed: '+str(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

dataset = sys.argv[1]
drop_prob = float(sys.argv[2])
repack_ratio = float(sys.argv[3])
bat_size = int(sys.argv[4])
embLen = int(sys.argv[5])

train_file = './data/intermediate/' + dataset + '/rm/train.data'
test_file = './data/intermediate/' + dataset + '/rm/test.data'
feature_file = './data/intermediate/' + dataset + '/rm/feature.txt'
type_file = './data/intermediate/' + dataset + '/rm/type.txt'
none_ind = utils.get_none_id(type_file)
print("None id:", none_ind)

word_size, pos_embedding_tensor = utils.initialize_embedding(feature_file, embLen)
_, type_size, _, _, _ = utils.load_corpus(train_file)
doc_size_test, _, feature_list_test, label_list_test, type_list_test = utils.load_corpus(test_file)

nocluster = noCluster.noCluster(embLen, word_size, type_size, drop_prob)
nocluster.load_state_dict(torch.load('./dumped_models/ffnn_dump_'+'_'.join(sys.argv[1:7])+'.pth'))

torch.cuda.set_device(0)
nocluster.cuda()
if_cuda = True

packer = pack.repack(repack_ratio, 20, if_cuda)
fl_t, of_t = packer.repack_eva(feature_list_test)

nocluster.eval()
scores = nocluster(fl_t, of_t)
ind = utils.calcInd(scores)
entropy = utils.calcEntropy(scores)
maxprob = utils.calcMaxProb(scores)

golden_list = []
predict_list = []

print(type(label_list_test[0]))
print(type(ind))

# vanilla prediction

for i in range(len(label_list_test)):
    golden_list.append(label_list_test[i][0])
    predict_list.append(ind.data[i])

filename = './case_study/' + dataset + '_case_study.txt'
file = open(filename, 'w')
file.write(str(predict_list)+'\n')
file.write(str(golden_list)+'\n')

f1score, recall, precision, val_f1 = utils.noCrossValidation(ind.data, entropy.data, label_list_test, ind.data, entropy.data, label_list_test, none_ind)
print('F1 = %.4f, recall = %.4f, precision = %.4f' % (f1score, recall, precision))
file.write('F1 = %.4f, recall = %.4f, precision = %.4f' % (f1score, recall, precision))
print(val_f1)

# max threshold
ndevF1, f1score, recall, precision, meanBestF1 = utils.CrossValidation_New(ind.data, maxprob.data, label_list_test, ind.data, maxprob.data, label_list_test, none_ind, thres_type='max')
print('Max Thres \tF1 = %.4f, recall = %.4f, precision = %.4f, ndevF1 = %.4f, cdevF1 = %.4f' % (f1score, recall, precision, ndevF1, meanBestF1))

# entropy threshold
ndevF1, f1score, recall, precision, meanBestF1 = utils.CrossValidation_New(ind.data, entropy.data, label_list_test, ind.data, entropy.data, label_list_test, none_ind, thres_type='entropy')
print('Entropy Thres \tF1 = %.4f, recall = %.4f, precision = %.4f, ndevF1 = %.4f, cdevF1 = %.4f' % (f1score, recall, precision, ndevF1, meanBestF1))