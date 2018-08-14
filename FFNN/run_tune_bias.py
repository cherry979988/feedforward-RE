import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random
import sys
import model.utils as utils
import model.noCluster as noCluster
import model.pack as pack

zip = getattr(itertools, 'izip', zip)

if len(sys.argv) != 6:
    print('Usage: run.py -DATA -outputDropout(0.2) -inputDropout(0) -batchSize(20) -randomseed(1234)')
    exit(1)

SEED = int(sys.argv[5])
print('Using Random Seed: '+str(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

dataset = sys.argv[1]
train_file = './data/intermediate/' + dataset + '/rm/train.data'
dev_file = './data/intermediate/' + dataset + '/rm/dev.data'
test_file = './data/intermediate/' + dataset + '/rm/test.data'

feature_file = './data/intermediate/' + dataset + '/rm/feature.txt'
type_file = './data/intermediate/' + dataset + '/rm/type.txt'
none_ind = utils.get_none_id(type_file)
print("None id:", none_ind)
embLen = 50

# tunable prams
drop_prob = float(sys.argv[2])
repack_ratio = float(sys.argv[3])
bat_size = int(sys.argv[4])

word_size, pos_embedding_tensor = utils.initialize_embedding(feature_file, embLen)

doc_size, type_size, feature_list, label_list, type_list = utils.load_corpus(test_file)

# doc_size_test, _, feature_list_test, label_list_test, type_list_test = utils.load_corpus(test_file)

# doc_size_dev, _, feature_list_dev, label_list_dev, type_list_dev = utils.load_corpus(dev_file)

nocluster = noCluster.noCluster(embLen, word_size, type_size, drop_prob)
nocluster.load_state_dict(torch.load('ffnn_dump.pth'))
nocluster.freeze_params()

# optimizer = utils.sgd(nocluster.parameters(), lr=0.025)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, nocluster.parameters()), lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

torch.cuda.set_device(0)
nocluster.cuda()
if_cuda = True

best_f1 = float('-inf')
best_recall = 0
best_precision = 0
best_meanBestF1 = float('-inf')
packer = pack.repack(repack_ratio, 20, if_cuda)

shuffle_type_list, shuffle_feature_list, shuffle_label_list = utils.shuffle_data_with_type(type_list, feature_list, label_list)
clean_dev_size = int(0.1 * len(shuffle_type_list))

cdev_type_list = shuffle_type_list[:clean_dev_size]
cdev_feature_list = shuffle_feature_list[:clean_dev_size]
cdev_label_list = shuffle_label_list[:clean_dev_size]
test_type_list = shuffle_type_list[clean_dev_size:]
test_feature_list = shuffle_feature_list[clean_dev_size:]
test_label_list = shuffle_label_list[clean_dev_size:]

fl_t, of_t = packer.repack_eva(test_feature_list)
fl_cd, of_cd = packer.repack_eva(cdev_feature_list)

for epoch in range(200):
    print("epoch: " + str(epoch))
    nocluster.train()
    sf_tp, sf_fl = utils.shuffle_data(cdev_type_list, cdev_feature_list)
    for b_ind in range(0, len(sf_tp), bat_size):
        nocluster.zero_grad()
        if b_ind + bat_size > len(sf_tp):
            b_eind = len(sf_tp)
        else:
            b_eind = b_ind + bat_size
        t_t, fl_rt1, fl_rt2, fl_dt, off_dt = packer.repack(sf_fl[b_ind: b_eind], sf_tp[b_ind: b_eind])
        loss = nocluster.NLL_loss(t_t, fl_rt1, fl_rt2, fl_dt, off_dt, 2)
        loss.backward()
        nn.utils.clip_grad_norm(nocluster.parameters(), 5)
        optimizer.step()

    # evaluation mode
    nocluster.eval()
    scores = nocluster(fl_t, of_t)
    ind = utils.calcInd(scores)
    entropy = utils.calcEntropy(scores)

    scores_cdev = nocluster(fl_cd, of_cd)
    ind_cdev = utils.calcInd(scores_cdev)
    entropy_cdev = utils.calcEntropy(scores_cdev)

    f1score, recall, precision, meanBestF1 = utils.noCrossValidation(ind_cdev.data, entropy_cdev.data, cdev_label_list, ind.data, entropy.data, test_label_list, none_ind)
    # f1score, recall, precision, meanBestF1 = utils.eval_score(ind_dev.data, entropy_dev.data, label_list_dev, ind.data, entropy.data, label_list_test, none_ind)
    scheduler.step(meanBestF1)

    print('F1 = %.4f, recall = %.4f, precision = %.4f, cdev f1 = %.4f)' %
          (f1score,
           recall,
           precision,
           meanBestF1))
    if meanBestF1 > best_meanBestF1:
        best_f1 = f1score
        best_recall = recall
        best_precision = precision
        best_meanBestF1 = meanBestF1

print('Best result: ')
print('F1 = %.4f, recall = %.4f, precision = %.4f, cdev f1 = %.4f)' %
      (best_f1,
       best_recall,
       best_precision,
       best_meanBestF1))

utils.save_tune_log(dataset, drop_prob, repack_ratio, bat_size, best_f1, best_recall, best_precision, best_meanBestF1)