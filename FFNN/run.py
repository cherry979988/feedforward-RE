import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random
import sys
import os
import model.utils as utils
import model.noCluster as noCluster
import model.pack as pack

zip = getattr(itertools, 'izip', zip)

if len(sys.argv) != 8:
    print('Usage: run.py -DATA -outputDropout(0.2) -inputDropout(0) -batchSize(20) -embLen(50) -randomseed(1234) -info')
    exit(1)

SEED = int(sys.argv[6])
print('Using Random Seed: '+str(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

dataset = sys.argv[1]
train_file = './data/intermediate/' + dataset + '/rm/train_x_corpus.txt'
train_label_file = './data/intermediate/' + dataset + '/rm/train_y_corpus.txt'
dev_file = './data/intermediate/' + dataset + '/rm/dev.data'
test_file = './data/intermediate/' + dataset + '/rm/test.data'

feature_file = './data/intermediate/' + dataset + '/rm/feature.txt'
type_file = './data/intermediate/' + dataset + '/rm/type.txt'
type_file_test = './data/intermediate/' + dataset + '/rm/type_test.txt'
none_ind = utils.get_none_id(type_file)

label_distribution = utils.get_distribution(type_file)
label_distribution_test = utils.get_distribution(type_file_test)
# print(label_distribution)
print("None id:", none_ind)

# tunable prams
drop_prob = float(sys.argv[2])
repack_ratio = float(sys.argv[3])
bat_size = int(sys.argv[4])
embLen = int(sys.argv[5])
info = sys.argv[7]

word_size, pos_embedding_tensor = utils.initialize_embedding(feature_file, embLen)

doc_size, type_size, feature_list, label_list, type_list = utils.load_corpus_new(train_file, train_label_file)

doc_size_test, _, feature_list_test, label_list_test, type_list_test = utils.load_corpus(test_file)

doc_size_dev, _, feature_list_dev, label_list_dev, type_list_dev = utils.load_corpus(dev_file)

nocluster = noCluster.noCluster(embLen, word_size, type_size, drop_prob, label_distribution, label_distribution_test)
print('embLen, word_size, type_size: ', embLen, word_size, type_size)

nocluster.load_word_embedding(pos_embedding_tensor)

# nocluster.load_neg_embedding(neg_embedding_tensor)

# optimizer = utils.sgd(nocluster.parameters(), lr=0.025)
optimizer = optim.SGD(nocluster.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
print('Total parameters: ', sum(p.numel() for p in nocluster.parameters() if p.requires_grad))

torch.cuda.set_device(0)
nocluster.cuda()
if_cuda = True

best_f1 = float('-inf')
best_recall = 0
best_precision = 0
best_meanBestF1 = float('-inf')
packer = pack.repack(repack_ratio, 20, if_cuda)

fl_t, of_t = packer.repack_eva(feature_list_test)
fl_d, of_d = packer.repack_eva(feature_list_dev)

for epoch in range(200):
    print("epoch: " + str(epoch))
    nocluster.train()
    sf_tp, sf_fl = utils.shuffle_data(type_list, feature_list)
    # print('#sample:',len(sf_tp))
    for b_ind in range(0, len(sf_tp), bat_size):
        nocluster.zero_grad()
        if b_ind + bat_size > len(sf_tp):
            b_eind = len(sf_tp)
        else:
            b_eind = b_ind + bat_size
        t_t, fl_rt1, fl_rt2, fl_dt, off_dt, scope = packer.repack(sf_fl[b_ind: b_eind], sf_tp[b_ind: b_eind])
        loss = nocluster.NLL_loss(t_t, fl_rt1, fl_rt2, fl_dt, off_dt, 2, scope)
        loss.backward()
        nn.utils.clip_grad_norm(nocluster.parameters(), 5)
        optimizer.step()
    # evaluation mode
    nocluster.eval()
    scores = nocluster(fl_t, of_t, 'test')
    ind = utils.calcInd(scores)
    entropy = utils.calcEntropy(scores)

    scores_dev = nocluster(fl_d, of_d, 'dev')
    ind_dev = utils.calcInd(scores_dev)
    entropy_dev = utils.calcEntropy(scores_dev)

    f1score, recall, precision, meanBestF1, _, _, _, _ = utils.noCrossValidation(ind_dev.data, entropy_dev.data, label_list_dev, ind.data, entropy.data, label_list_test, none_ind)
    # f1score, recall, precision, meanBestF1 = utils.eval_score(ind_dev.data, entropy_dev.data, label_list_dev, ind.data, entropy.data, label_list_test, none_ind)
    scheduler.step(meanBestF1)

    print('F1 = %.4f, recall = %.4f, precision = %.4f, val f1 = %.4f)' %
          (f1score,
           recall,
           precision,
           meanBestF1))
    if meanBestF1 > best_meanBestF1:
        best_f1 = f1score
        best_recall = recall
        best_precision = precision
        best_meanBestF1 = meanBestF1
        if not os.path.exists('./dumped_models/%s' % info):
            os.mkdir('./dumped_models/%s' % info)
        torch.save(nocluster.state_dict(), './dumped_models/%s/ffnn_dump_' % info +'_'.join(sys.argv[1:7])+'.pth')

print('Best result: ')
print('F1 = %.4f, recall = %.4f, precision = %.4f, val f1 = %.4f)' %
      (best_f1,
       best_recall,
       best_precision,
       best_meanBestF1))

utils.save_tune_log(dataset, drop_prob, repack_ratio, bat_size, embLen, best_f1, best_recall, best_precision, best_meanBestF1)
