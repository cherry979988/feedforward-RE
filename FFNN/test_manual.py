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
    print('Usage: run.py -DATA -outputDropout(0.2) -inputDropout(0) -batchSize(20) -embLen(50) -bagWeighting(none/ave/att) -randomseed(1234) -info')
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
bagWeighting = sys.argv[6]
info = sys.argv[7]

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
nocluster.load_state_dict(torch.load('./dumped_models/KBP_bias_eq_log_distrib_orig_hyper/ffnn_dump_'+'_'.join(sys.argv[1:7])+'.pth'))

torch.cuda.set_device(0)
nocluster.cuda()
if_cuda = True

packer = pack.repack(repack_ratio, 20, if_cuda)

print('in the order of: train, dev, test...\n')
for file in [train_file, dev_file, test_file]:
    doc_size_test, _, feature_list_test, label_list_test, type_list_test = utils.load_corpus(file)

    fl_t, of_t = packer.repack_eva(feature_list_test)

    nocluster.eval()
    scores = nocluster(fl_t, of_t, 'test')
    ind = utils.calcInd(scores)
    entropy = utils.calcEntropy(scores)
    maxprob = utils.calcMaxProb(scores)

    golden_list = []
    predict_list = []

    # vanilla prediction

    for i in range(len(label_list_test)):
        golden_list.append(label_list_test[i][0])
        predict_list.append(ind.data[i])

    f1score, recall, precision, val_f1, pn_precision, nprecision, nrecall, accuracy = utils.noCrossValidation(ind.data, entropy.data, label_list_test, ind.data, entropy.data, label_list_test, none_ind)
    print('F1 = %.4f, recall = %.4f, precision = %.4f, pn_precision = %.4f, accuracy = %.4f' % (f1score, recall, precision, pn_precision, accuracy))
    print('NoneType precision: ', nprecision, ' recall: ', nrecall)

    if file == test_file:
        print(len(golden_list))
        filename = './case_study/' + dataset + '_case_study.txt'
        file = open(filename, 'w')
        file.write(str([predict_list[i] for i in range(len(predict_list))])+'\n')
        file.write(str([golden_list[i] for i in range(len(golden_list))])+'\n')
        file.write('F1 = %.4f, recall = %.4f, precision = %.4f, accuracy = %.4f' % (f1score, recall, precision, accuracy))
        file.close()

        intermediate_emb_bag = nocluster.men_embedding.data.tolist()
        if not os.path.exists('./case_study/%s' % info):
            os.mkdir('./case_study/%s' % info)
        filename = './case_study/%s/'%info + dataset + '_embedding_test.mat'
        #file = open(filename, 'w')
        #for i in range(len(intermediate_emb_bag)):
        #    file.write(str(list(intermediate_emb_bag[i])))
        #file.close()
        sio.savemat(filename, {'inter_emb': intermediate_emb_bag})


filename = './case_study/' + dataset + '_case_study.txt'
file = open(filename, 'a')
# max threshold
ndevF1, f1score, recall, precision, meanBestF1, pn_precision, bestThres, acc = utils.CrossValidation_New(ind.data, maxprob.data, label_list_test, ind.data, maxprob.data, label_list_test, none_ind, thres_type='max')
print('Max Thres \tF1 = %.4f, recall = %.4f, precision = %.4f, ndevF1 = %.4f, cdevF1 = %.4f, pn_precision = %.4f' % (f1score, recall, precision, ndevF1, meanBestF1, pn_precision))
print('Accuracy: %.4f' % acc)
pre_ind_cutoff = utils.eval_score_with_thres(ind.data, maxprob.data, label_list_test,none_ind,bestThres,thres_type='max')
file.write(str([pre_ind_cutoff[i] for i in range(len(pre_ind_cutoff))])+'\n')

# entropy threshold
ndevF1, f1score, recall, precision, meanBestF1, pn_precision, bestThres, acc = utils.CrossValidation_New(ind.data, entropy.data, label_list_test, ind.data, entropy.data, label_list_test, none_ind, thres_type='entropy')
print('Entropy Thres \tF1 = %.4f, recall = %.4f, precision = %.4f, ndevF1 = %.4f, cdevF1 = %.4f, pn_precision = %.4f' % (f1score, recall, precision, ndevF1, meanBestF1, pn_precision))
print('Accuracy: %.4f' % acc)
pre_ind_cutoff = utils.eval_score_with_thres(ind.data, entropy.data, label_list_test,none_ind,bestThres,thres_type='entropy')
file.write(str([pre_ind_cutoff[i] for i in range(len(pre_ind_cutoff))])+'\n')
file.close()

# bias adjustment
doc_size_test, _, feature_list_test, label_list_test, type_list_test = utils.load_corpus(test_file)
#data = [[feature_list_test[i], label_list_test[i]] for i in range(len(feature_list_test))]
#random.shuffle(data)
labels, feature = utils.shuffle_data(label_list_test, feature_list_test)
eva_n = int(0.1 * len(label_list_test))

eva_label = labels[:eva_n]
eva_label_distribution = utils.get_distribution_from_list(eva_label)
print(eva_label_distribution)
print(label_distribution_test)
#eva_feature = feature[:eva_n]

test_label = labels[eva_n:]
test_feature = feature[eva_n:]

fl_t, of_t = packer.repack_eva(test_feature)

nocluster.eval()
scores = nocluster.test_with_bias(fl_t, of_t, eva_label_distribution)
#scores = nocluster(fl_t, of_t, 'test')
ind = utils.calcInd(scores)
# golden_list = []
# predict_list = []
#
# for i in range(len(test_label)):
#     golden_list.append(test_label[i][0])
#     predict_list.append(ind.data[i])
f1score, recall, precision, val_f1, pn_precision, nprecision, nrecall, accuracy = utils.noCrossValidation(ind.data, entropy.data, test_label, ind.data, entropy.data, test_label, none_ind)
print('F1 = %.4f, recall = %.4f, precision = %.4f, pn_precision = %.4f, accuracy = %.4f' % (f1score, recall, precision, pn_precision, accuracy))
print('NoneType precision: ', nprecision, ' recall: ', nrecall)

# python3 FFNN/test_manual.py KBP 0.5 0.1 80 50 none 4 none