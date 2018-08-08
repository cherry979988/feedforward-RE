import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

class EmbeddingBag_self(nn.Module):

    def __init__(self, num_embeddings, embedding_dim,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 mode='mean'):
        super(EmbeddingBag_self, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def forward(self, input, offsets=None):
        out1 = F.embedding_bag(self.weight, input, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, 'mean')
        out2 = F.embedding_bag(self.weight, input, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, 'sum')
        #return out1
        return torch.cat([out1, out2], 1)

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        s += ', mode={mode}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class EmbeddingBag_Mean_n_Max(nn.Module):

    def __init__(self, num_embeddings, embedding_dim,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 mode='mean'):
        super(EmbeddingBag_Mean_n_Max, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def forward(self, input, offsets=None):
        #print(input)
        #print(offsets)
        out1 = F.embedding_bag(input, self.weight, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, 'mean')
        out2 = F.embedding_bag(input, self.weight, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, 'max')
        #print(out1)
        return out1
        # return torch.cat([out1, out2], 1)

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        s += ', mode={mode}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)