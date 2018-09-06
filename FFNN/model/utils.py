import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random
import time
import os
import pickle

zip = getattr(itertools, 'izip', zip)


def get_none_id(type_filename):
    with open(type_filename, encoding='utf-8') as type_file:
        for line in type_file:
            ls = line.strip().split()
            if ls[0] == "None":
                return int(ls[1])


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def log_sum_exp(vec):
    # vec: B * L * M
    # output: B * 1 * M
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx)  # B * 1 * M
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1))  # B * 1 * M


def soft_max(vec, mask):
    batch_size = vec.size(0)
    _, idx = torch.max(vec, 1)  # B * 1
    idx = idx.view(batch_size, 1)
    max_score = torch.gather(vec, 1, idx)  # B * 1
    max_score = max_score.view(batch_size, 1)
    exp_score = torch.exp(vec - max_score.expand_as(vec))  # B * L
    exp_score = exp_score * mask  # B * L
    exp_score_sum = torch.sum(exp_score, 1).view(batch_size, 1).expand_as(exp_score)
    prob_score = exp_score / exp_score_sum
    return prob_score


def load_embedding(emb_file, randIni=True):
    fin = open(emb_file, 'r')
    line = fin.readline().split(' ')
    size = int(line[0])
    embLen = int(line[1])
    if randIni:
        bias = np.sqrt(3.0 / embLen)
        pos_embedding_tensor = torch.rand(size, embLen) * 2 * bias - bias
        neg_embedding_tensor = torch.FloatTensor(size, embLen).zero_()
    else:
        pos_embedding_array = list()
        neg_embedding_array = list()
        for ind in range(size):
            line = fin.readline()
            if line.isspace():
                break
            line = line.split(' ')
            vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line)))
            pos_embedding_array.append(vector)
        fin.readline()
        for ind in range(size):
            line = fin.readline()
            if line.isspace():
                break
            line = line.split(' ')
            vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line)))
            neg_embedding_array.append(vector)
        pos_embedding_tensor = torch.FloatTensor(np.asarray(pos_embedding_array))
        neg_embedding_tensor = torch.FloatTensor(np.asarray(neg_embedding_array))
    return size, embLen, pos_embedding_tensor, neg_embedding_tensor


def initialize_embedding(feature_file, embLen):
    fin = open(feature_file, 'r')
    lines = fin.readlines()
    size = len(lines)
    bias = np.sqrt(3.0 / embLen)
    embedding_tensor = torch.rand(size, embLen) * 2 * bias - bias
    return size, embedding_tensor


def load_corpus(corpus, if_cuda=True):
    fin = open(corpus, 'r')
    line = fin.readline()
    size = int(line)
    feature_list = list()
    label_list = list()
    type_size = -1
    for line in fin:
        if line.isspace():
            break
        line = line.split('\t')
        fv = torch.LongTensor()
        feature_list.append(torch.LongTensor(list(map(lambda t: int(t), line[2].split(' ')))))
        label_vec = list(map(lambda t: int(t), line[4].split(' ')))
        tmp = max(label_vec)
        if type_size < tmp:
            type_size = tmp
        #if if_cuda:
        #    label_list.append(torch.cuda.LongTensor(label_vec))
        #else:
        label_list.append(torch.LongTensor(label_vec))
    type_size = type_size + 1
    type_list = list()
    for label in label_list:
        type_vec = torch.FloatTensor(type_size).zero_()
        type_vec[label] = 1
        type_vec = type_vec.view(1, -1)
        type_list.append(type_vec)
    return size, type_size, feature_list, label_list, type_list


def load_qa_corpus(corpus):
    fin = open(corpus, 'r')
    size = 0
    feature_list = list()
    for line in fin:
        if line.isspace():
            break
        size += 1
        line = line.strip('\r\n').split('\t')
        feature_list.append(torch.LongTensor(list(map(lambda t: int(t), line[1].split(',')))))
    return size, feature_list


def load_question_info(mention_question_file):
    pos_qapairs = []
    pos_qapair_to_question = {}
    question_to_qapairs = {}
    with open(mention_question_file, 'r') as fin:
        for line in fin:
            if line.isspace():
                break
            line = line.strip('\r\n').split('\t')
            mid = int(line[0])
            qid = int(line[1])
            if float(line[2]) == 1.0:
                pos_qapairs.append(mid)
                pos_qapair_to_question[mid] = qid
                if qid in question_to_qapairs:
                    pos_mids = question_to_qapairs[qid][0]
                    pos_mids.append(mid)
                    question_to_qapairs[qid] = (pos_mids, question_to_qapairs[qid][1])
                else:
                    question_to_qapairs[qid] = ([mid], [])
            else:
                if qid in question_to_qapairs:
                    neg_mids = question_to_qapairs[qid][1]
                    neg_mids.append(mid)
                    question_to_qapairs[qid] = (question_to_qapairs[qid][0], neg_mids)
                else:
                    question_to_qapairs[qid] = ([], [mid])
    return pos_qapairs, pos_qapair_to_question, question_to_qapairs


def shuffle_data(ori_labels, ori_corpus):
    assert (len(ori_labels) == len(ori_corpus))
    index_shuf = list(range(len(ori_labels)))
    random.shuffle(index_shuf)
    labels = [ori_labels[i] for i in index_shuf]
    corpus = [ori_corpus[i] for i in index_shuf]
    return labels, corpus

def shuffle_data_with_type(ori_labels, ori_corpus, ori_types):
    assert (len(ori_labels) == len(ori_corpus) and len(ori_labels) == len(ori_types))
    index_shuf = list(range(len(ori_labels)))
    random.shuffle(index_shuf)
    labels = [ori_labels[i] for i in index_shuf]
    corpus = [ori_corpus[i] for i in index_shuf]
    types = [ori_types[i] for i in index_shuf]
    return labels, corpus, types

def calcEntropy(batch_scores):
    # input: B * L
    # output: B
    batch_probs = nn.functional.softmax(batch_scores)
    return torch.sum(batch_probs * torch.log(batch_probs), 1).neg()


def calcInd(batch_probs):
    # input: B * L
    # output: B
    _, ind = torch.max(batch_probs, 1)
    return ind

def calcMaxProb(batch_scores):
    batch_probs = nn.functional.softmax(batch_scores)
    prob, _ = torch.max(batch_probs, 1)
    return prob

def dropout(input_tensor, ratio):
    keep_vec = torch.ge(torch.rand(input_tensor.size()), ratio)
    return input_tensor[keep_vec]


def resample(input_tensor, resample_num):
    resample_ind = torch.Tensor(resample_num).uniform_(0, input_tensor.size(0) - 1).long()
    return input_tensor[resample_ind].view(-1)


def clip_grad(parameters, max_value):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        mask = p.grad.data.gt(max_value)
        p.grad.data[mask] = max_value
        mask = p.grad.data.lt(-max_value)
        p.grad.data[mask] = -max_value


def CrossValidation(pre_ind, pre_entropy, true_ind, noneInd, ratio=0.1, cvnum=100):
    f1score = 0.0
    recall = 0.0
    precision = 0.0
    meanBestF1 = 0.0
    valSize = int(np.floor(ratio * len(pre_ind)))
    data = [[pre_ind[ind], pre_entropy[ind], true_ind[ind]] for ind in range(0, len(pre_ind))]
    for cvind in range(cvnum):
        random.shuffle(data)
        val = data[0:valSize]
        eva = data[valSize:]

        # find best threshold
        max_ent = max(val, key=lambda t: t[1])[1]
        min_ent = min(val, key=lambda t: t[1])[1]
        stepSize = (max_ent - min_ent) / 100
        thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
        ofInterest = 0
        for ins in val:
            if ins[2][0] != noneInd:
                ofInterest += 1
        bestThreshold = float('nan')
        bestF1 = float('-inf')
        for threshold in thresholdList:
            corrected = 0
            predicted = 0
            for ins in val:
                if ins[1] < threshold and ins[0] != noneInd:
                    predicted += 1
                    if ins[0] == ins[2][0]:
                        corrected += 1
            curF1 = 2.0 * corrected / (ofInterest + predicted)
            if curF1 > bestF1:
                bestF1 = curF1
                bestThreshold = threshold
        meanBestF1 += bestF1
        ofInterest = 0
        corrected = 0
        predicted = 0
        for ins in eva:
            if ins[2][0] != noneInd:
                ofInterest += 1
            if ins[1] < bestThreshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2][0]:
                    corrected += 1
        f1score += (2.0 * corrected / (ofInterest + predicted))
        recall += (1.0 * corrected / ofInterest)
        precision += (1.0 * corrected / (predicted + 0.00001))

    meanBestF1 /= cvnum
    f1score /= cvnum
    recall /= cvnum
    precision /= cvnum
    return f1score, recall, precision, meanBestF1


def eval_score(pre_ind_dev, pre_entropy_dev, true_ind_dev, pre_ind_test, pre_entropy_test, true_ind_test, noneInd):
    '''
    :param pre_ind_dev:
    :param pre_entropy_dev:
    :param true_ind_dev:
    :param pre_ind_test:
    :param pre_entropy_test:
    :param true_ind_test:
    :param noneInd:
    :param ratio: only a ratio of input dev will be used to tune threshold
    :return: test_f1, test_precision, test_recall, dev_f1
    '''

    val = [[pre_ind_dev[ind], pre_entropy_dev[ind], true_ind_dev[ind]] for ind in range(0, len(pre_ind_dev))]
    eva = [[pre_ind_test[ind], pre_entropy_test[ind], true_ind_test[ind]] for ind in range(0, len(pre_ind_test))]

    # find best threshold
    max_ent = max(val, key=lambda t: t[1])[1]
    min_ent = min(val, key=lambda t: t[1])[1]
    stepSize = (max_ent - min_ent) / 100
    thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
    ofInterest = 0
    for ins in val:
        if ins[2][0] != noneInd:
            ofInterest += 1
    bestThreshold = float('nan')
    bestF1 = float('-inf')
    for threshold in thresholdList:
        corrected = 0
        predicted = 0
        for ins in val:
            if ins[1] < threshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2][0]:
                    corrected += 1
        curF1 = 2.0 * corrected / (ofInterest + predicted)
        if curF1 > bestF1:
            bestF1 = curF1
            bestThreshold = threshold

    # tuning complete, evaluate on test set
    meanBestF1 = bestF1
    ofInterest = 0
    corrected = 0
    predicted = 0
    for ins in eva:
        if ins[2][0] != noneInd:
            ofInterest += 1
        if ins[1] < bestThreshold and ins[0] != noneInd:
            predicted += 1
            if ins[0] == ins[2][0]:
                corrected += 1
    f1score = (2.0 * corrected / (ofInterest + predicted))
    recall = (1.0 * corrected / ofInterest)
    precision = (1.0 * corrected / (predicted + 0.00001))

    return f1score, recall, precision, meanBestF1

def noCrossValidation(pre_ind_dev, pre_entropy_dev, true_ind_dev, pre_ind_test, pre_entropy_test, true_ind_test, noneInd):
    # direct evaluation on test-set, no threshold tuning!
    # designed to figure out the diff. between FFNN and CoType
    # val_f1 is on dev set, p/r/f1 is on test set

    # dev set
    f1score = 0.0
    recall = 0.0
    precision = 0.0
    data = [[pre_ind_dev[ind], pre_entropy_dev[ind], true_ind_dev[ind]] for ind in range(0, len(pre_ind_dev))]

    max_ent = max(data, key=lambda t: t[1])[1]
    threshold = max_ent

    ofInterest = 0
    for ins in data:
        if ins[2][0] != noneInd:
            ofInterest += 1
    corrected = 0
    predicted = 0
    for ins in data:
        if ins[0] != noneInd:
            predicted += 1
            if ins[0] == ins[2][0]:
                corrected += 1
    f1score = (2.0 * corrected / (ofInterest + predicted))
    recall = (1.0 * corrected / ofInterest)
    precision = (1.0 * corrected / (predicted + 1e-8))
    val_f1 = f1score

    # test set
    data = [[pre_ind_test[ind], pre_entropy_test[ind], true_ind_test[ind]] for ind in range(0, len(pre_ind_test))]

    ofInterest = 0
    for ins in data:
        if ins[2][0] != noneInd:
            ofInterest += 1
    corrected = 0
    predicted = 0
    for ins in data:
        if ins[0] != noneInd:
            predicted += 1
            if ins[0] == ins[2][0]:
                corrected += 1
    f1score = (2.0 * corrected / (ofInterest + predicted))
    recall = (1.0 * corrected / ofInterest)
    precision = (1.0 * corrected / (predicted + 1e-8))

    corrected = 0
    for ins in data:
        if ins[0] == noneInd and ins[2][0] == noneInd:
            corrected += 1
        if ins[0] != noneInd and ins[2][0] != noneInd:
            corrected += 1
    pn_precision = corrected / len(pre_ind_test) # positive_negative precision

    corrected = 0.0
    predicted = 0.0
    ofInterest = 0.0
    for ins in data:
        if ins[0] == noneInd and ins[2][0] == noneInd:
            corrected += 1
        if ins[2][0] == noneInd:
            ofInterest += 1
        if ins[0] == noneInd:
            predicted += 1
    # print('NoneType precision: ', corrected/predicted, ' recall: ', corrected/ofInterest)

    return f1score, recall, precision, val_f1, pn_precision, corrected/predicted, corrected/ofInterest

def CrossValidation_New(pre_ind_ndev, pre_entropy_ndev, true_ind_ndev, pre_ind, pre_entropy, true_ind, noneInd, thres_type='max', ratio=0.1, cvnum=100):
    # > for 'max' and < for 'entropy'
    if thres_type == 'max':
        SIGN = 1
    else:
        SIGN = -1
        
    # calculate the f1 without threshold on ndev set
    f1score = 0.0
    recall = 0.0
    precision = 0.0
    data = [[pre_ind_ndev[ind], pre_entropy_ndev[ind], true_ind_ndev[ind]] for ind in range(0, len(pre_ind_ndev))]

    max_ent = max(data, key=lambda t: t[1])[1]
    threshold = max_ent

    ofInterest = 0
    for ins in data:
        if ins[2][0] != noneInd:
            ofInterest += 1
    corrected = 0
    predicted = 0
    for ins in data:
        if ins[0] != noneInd:
            predicted += 1
            if ins[0] == ins[2][0]:
                corrected += 1
    f1score = (2.0 * corrected / (ofInterest + predicted))
    recall = (1.0 * corrected / ofInterest)
    precision = (1.0 * corrected / (predicted + 1e-8))
    ndev_f1 = f1score

    # tune threshold (domain adaption).
    # can either be max thres or entropy thres; works in the same way.
    f1score = 0.0
    recall = 0.0
    precision = 0.0
    meanBestF1 = 0.0
    pn_precision = 0.0
    valSize = int(np.floor(ratio * len(pre_ind)))
    data = [[pre_ind[ind], pre_entropy[ind], true_ind[ind]] for ind in range(0, len(pre_ind))]
    for cvind in range(cvnum):
        random.shuffle(data)
        val = data[0:valSize]
        eva = data[valSize:]

        # find best threshold
        max_ent = max(val, key=lambda t: t[1])[1]
        min_ent = min(val, key=lambda t: t[1])[1]
        stepSize = (max_ent - min_ent) / 100
        thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
        ofInterest = 0
        for ins in val:
            if ins[2][0] != noneInd:
                ofInterest += 1
        bestThreshold = float('nan')
        bestF1 = float('-inf')
        for threshold in thresholdList:
            corrected = 0
            predicted = 0
            for ins in val:
                if SIGN * ins[1] > SIGN * threshold and ins[0] != noneInd:
                    predicted += 1
                    if ins[0] == ins[2][0]:
                        corrected += 1
            curF1 = 2.0 * corrected / (ofInterest + predicted)
            if curF1 > bestF1:
                bestF1 = curF1
                bestThreshold = threshold
        meanBestF1 += bestF1
        ofInterest = 0
        corrected = 0
        predicted = 0
        for ins in eva:
            if ins[2][0] != noneInd:
                ofInterest += 1
            if SIGN * ins[1] > SIGN * bestThreshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2][0]:
                    corrected += 1
        f1score += (2.0 * corrected / (ofInterest + predicted))
        recall += (1.0 * corrected / ofInterest)
        precision += (1.0 * corrected / (predicted + 0.00001))

        corrected = 0
        for ins in eva:
            if (ins[0] == noneInd or SIGN * ins[1] < SIGN * bestThreshold) and ins[2][0] == noneInd:
                corrected += 1
            if (ins[0] != noneInd and SIGN * ins[1] >= SIGN * bestThreshold) and ins[2][0] != noneInd:
                corrected += 1
        pn_precision += corrected / len(eva) # positive_negative precision

    meanBestF1 /= cvnum
    f1score /= cvnum
    recall /= cvnum
    precision /= cvnum
    pn_precision /= cvnum
    return ndev_f1, f1score, recall, precision, meanBestF1, pn_precision
    
def save_tune_log(dataset, drop_prob, repack_ratio, bat_size, embLen, f1, recall, precision, val_f1):
    if os.path.isfile('tune_log.pkl'):
        with open('tune_log.pkl', 'rb') as f:
            d = pickle.load(f)
    else:
        d = dict()
    d[(dataset, drop_prob, repack_ratio, bat_size, embLen)] = val_f1
    with open('tune_log.pkl', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    f = open('tune_full_log.txt', 'a+')
    f.write("Dataset: %s Drop_prob: %s Repack_ratio: %s Bat_size: %s embLen: %s\n" % (dataset, drop_prob, repack_ratio, bat_size, embLen))
    f.write("F1: %s Recall %s Precision %s Val_f1 %s\n" % (f1, recall, precision, val_f1))
    f.write("Time stamp: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    f.write("\n===\n")

def save_tune_log_cv(dataset, drop_prob, repack_ratio, bat_size, f1, recall, precision, cdev_f1, ndev_f1=None, info=None):
    if os.path.isfile('tune_log.pkl'):
        with open('tune_log.pkl', 'rb') as f:
            d = pickle.load(f)
    else:
        d = dict()
    d[(dataset, drop_prob, repack_ratio, bat_size)] = ndev_f1
    with open('tune_log.pkl', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    f = open('tune_full_log.txt', 'a+')
    f.write("Dataset: %s Drop_prob: %s Repack_ratio: %s Bat_size: %s\n" % (dataset, drop_prob, repack_ratio, bat_size))
    f.write("F1: %s Recall: %s Precision: %s Cdev_f1: %s" % (f1, recall, precision, cdev_f1))
    if ndev_f1 != None:
        f.write(" Ndev_f1: %s\n") % (ndev_f1)
    else:
        f.write("\n")
    if info != None:
        f.write("Info: " + info)
    f.write("Time stamp: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    f.write("\n===\n")

