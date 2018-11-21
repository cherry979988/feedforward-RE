import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import itertools
import numpy as np
import random
import model.nce as nce
import model.object as obj

class noCluster(nn.Module):
    def __init__(self, emblen, word_size, type_size, drop_prob, label_distribution, label_distribution_test, if_average=False):
        super(noCluster, self).__init__()
        self.emblen = emblen
        self.word_size = word_size
        self.type_size = type_size

        self.word_emb = nn.Embedding(word_size, emblen)
        self.word_emb_bag = nn.EmbeddingBag(word_size, emblen)
        self.word_embedding = self.word_emb.weight
        self.word_emb_bag.weight = self.word_embedding

        self.linear = nn.Linear(emblen, type_size, bias=False)
        self.linear.weight.data.zero_()

        self.label_distribution = label_distribution
        self.distribution_tensor = torch.log(torch.autograd.Variable(torch.cuda.FloatTensor(self.label_distribution), requires_grad=False))
        self.label_distribution_test = label_distribution_test
        self.distribution_tensor_test = torch.log(torch.autograd.Variable(torch.cuda.FloatTensor(self.label_distribution_test), requires_grad=False))

        # self.neg_word = nce.NCE_loss(word_size, emblen)

        self.crit = obj.partCE(if_average=if_average)
        self.drop_prob = drop_prob
        # self.crit = obj.softCE_S(if_average=if_average)
        # self.crit = obj.softCE(if_average=if_average)
        # self.crit = obj.softKL(if_average=if_average)


    def load_word_embedding(self, pre_embeddings):
        self.word_embedding = nn.Parameter(pre_embeddings)
        self.word_emb.weight = self.word_embedding
        self.word_emb_bag.weight = self.word_embedding

    # def load_neg_embedding(self, pre_embeddings):
        # self.neg_word.load_neg_embedding(pre_embeddings)

    def load_linear_weights(self, linear_weights):
        self.linear.weight = nn.Parameter(linear_weights)

    def NLL_loss(self, typeTensor, resampleFeature1, resampleFeature2, feaDrop, offsetDrop, neg_sample, scope):
        self.typeTensor = typeTensor
        scores = self(feaDrop, offsetDrop, scope=scope)
        # batch_size = scores.size(0)
        pos_word = self.word_emb(resampleFeature1)
        loss = self.crit(scores, typeTensor)
        # loss = self.neg_word(pos_word, resampleFeature2, neg_sample, batch_size) + self.crit(scores, typeTensor)
        return loss

    def forward(self, feature_seq, offset_seq, type='train', scope=None):
        if type != 'train':
            men_embedding = self.word_emb_bag(feature_seq, offset_seq)
            self.men_embedding = men_embedding
            return self.linear(F.dropout(men_embedding, p=self.drop_prob, training=self.training))
        else:
            mem_embedding = self.word_emb_bag(feature_seq, offset_seq)
            _, type = torch.max(self.typeTensor, 1)
            bag_emb = []
            for i in range(len(scope)-1):
                #print(scope[i].data[0])
                #print(scope[i+1].data[0])
                emb = mem_embedding[scope[i]:scope[i+1]]
                scores = self.linear(F.dropout(emb, p=self.drop_prob, training=self.training))
                att_weight = F.softmax(scores[:, type[i]])
                bag_emb.append(torch.matmul(att_weight, emb)) # direction?
                # print(torch.stack(bag_emb))
            return self.linear(F.dropout(torch.stack(bag_emb), p=self.drop_prob, training=self.training))

        #if not type == 'test':
        #    return self.linear(F.dropout(men_embedding, p=self.drop_prob, training=self.training)) + self.distribution_tensor
        #else:
        #    return self.linear(F.dropout(men_embedding, p=self.drop_prob, training=self.training)) + self.distribution_tensor_test


    def freeze_params(self):
        # add bias to linear layer
        weight_original = self.linear.weight
        self.linear = nn.Linear(self.emblen, self.type_size, bias=True)
        self.linear.weight = weight_original

        #freeze params
        for params in self.parameters():
            params.requires_grad = False
        self.linear.bias.requires_grad = True
        # self.linear.weight.requires_grad = False
        # self.word_emb.weight.requires_grad = False
        # self.word_emb_bag.weight.requires_grad = False




