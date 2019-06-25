###
'''
April 2019
Code by: Arnaud Fickinger
'''
###

import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch
import math
# torch.manual_seed(42)
import pickle

from options import Options

opt = Options().parse()

# with open('/Users/arnaudfickinger/Documents/Research/harvard/one_test_theano.pkl', 'rb') as f:
#     theano_dic = pickle.load(f, encoding='latin1')


class Attention(nn.Module):
    def __init__(self, query_dim, alphabet_size, value_dim):
        super(Attention, self).__init__()
        self.query_weight = nn.Linear(alphabet_size, query_dim, bias=False)
        self.key_weight = nn.Linear(alphabet_size, query_dim, bias=False)
        self.value_weigt = nn.Linear(alphabet_size, value_dim, bias=False)
        self.key_dim = query_dim

    def forward(self, x):
        # print(x.shape)
        #x = x.reshape(-1, ) look at x shape
        value = self.value_weigt(x)
        key = self.key_weight(x)
        query = self.query_weight(x)
        attention = torch.matmul(query, torch.transpose(key, -1, -2))/math.sqrt(self.key_dim)
        attention = F.softmax(attention)
        result = torch.matmul(attention, value) #should be batch*seql*valuedim
        return result


class Position(nn.Module): #input and output have same size
    def __init__(self, interaction_dim, h_dim):
        super(Position, self).__init__()
        # self.interaction_dim = interaction_dim
        self.main = nn.Sequential(
            nn.Linear(interaction_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, interaction_dim)
        )

    def forward(self, x_att):
        # x_att = x_att.shape(-1, self.interaction_dim)
        return self.main(x_att)


class Normalize(nn.Module):
    def __init__(self, eps = 1e-6):
        super(Normalize, self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class Interaction(nn.Module):
    def __init__(self, query_dim, seq_length, value_dim, alphabet_size, h_dim, dropout):
        super(Interaction, self).__init__()
        self.attention = Attention(query_dim, alphabet_size, value_dim)
        self.position = Position(alphabet_size, h_dim)
        self.normalize1 = Normalize()
        self.normalize2 = Normalize()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print(x.shape)
        out = self.attention(x) #encoder: dim0= but dim1 longer
        out = self.dropout(out)
        # print(out.shape)
        out = self.normalize1(x+out)
        out2 = self.position(out)
        out2 = self.dropout(out2)
        return self.normalize2(out + out2)


class AVAE(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim, dec_h1_dim, dec_h2_dim, value_dim, query_dim, h_int_dim, dropout, mu_sparse=None, sigma_sparse=None, cw_inner_dimension=None, nb_patterns=None, hasTemperature = None, hasDictionary = None):
        super(AVAE, self).__init__()
        self.encoder = Encoder(latent_dim, sequence_length, alphabet_size, enc_h1_dim, enc_h2_dim, value_dim, query_dim, h_int_dim, dropout)
        self.decoder = Decoder(latent_dim, sequence_length, alphabet_size, dec_h1_dim, dec_h2_dim, value_dim, mu_sparse, sigma_sparse, cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, query_dim, h_int_dim, dropout)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = sample_diag_gaussian_original(mu, logsigma)
        px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l = self.decoder(z, x)
        return mu, logsigma, px_z, logpx_z, mu_W1, logsigma_W1, mu_b1, logsigma_b1, mu_W2, logsigma_W2, mu_b2, logsigma_b2, mu_W3, logsigma_W3, mu_b3, logsigma_b3, mu_S, logsigma_S, mu_C, logsigma_C, mu_l, logsigma_l


class Encoder(nn.Module):
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, value_dim, query_dim, h_int_dim, dropout):
        super(Encoder, self).__init__()
        self.sequence_length = sequence_length
        self.alphabet_size = alphabet_size
        self.value_dim = value_dim
        self.interaction = Interaction(query_dim, sequence_length, value_dim, alphabet_size, h_int_dim, dropout)
        if opt.attention_encoder:
            self.fc1 = nn.Linear(sequence_length*value_dim, h1_dim) #different parameters for every positions
        else:
            self.fc1 = nn.Linear(sequence_length * alphabet_size, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3_mu = nn.Linear(h2_dim, latent_dim)
        self.fc3_logsigma = nn.Linear(h2_dim, latent_dim)

    def forward(self, x):
        # print("x")
        # print(x.shape)
        # if x.shape[-1]!= self.sequence_length*self.alphabet_size:
        #     x = x.view(-1, self.sequence_length*self.alphabet_size)
        x = x.reshape(-1, self.sequence_length, self.alphabet_size)
        if opt.attention_encoder:
            interaction_representation = self.interaction(x)
        else:
            interaction_representation = x
        # print("int")
        # print(interaction_representation.shape)
        interaction_representation = interaction_representation.reshape(-1, self.sequence_length*self.value_dim)
        h1 = F.relu(self.fc1(interaction_representation))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc3_mu(h2)
        logsigma = self.fc3_logsigma(h2)
        return mu, logsigma

class Decoder(nn.Module):  # sparsity ideas of deep generative model for mutation paper
    def __init__(self, latent_dim, sequence_length, alphabet_size, h1_dim, h2_dim, value_dim, mu_sparse, sigma_sparse,
                 cw_inner_dimension, nb_patterns, hasTemperature, hasDictionary, query_dim, h_int_dim, dropout):
        super(Decoder, self).__init__()
        self.interaction = Interaction(query_dim, sequence_length, value_dim, alphabet_size, h_int_dim, dropout)
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.nb_patterns = nb_patterns
        self.mu_sparse = mu_sparse
        self.h2_dim = h2_dim
        self.logsigma_sparse = math.log(sigma_sparse)
        self.mu_W1 = nn.Parameter(torch.Tensor(latent_dim, h1_dim).normal_(0, 0.01))
        self.logsigma_W1 = nn.Parameter(torch.Tensor(latent_dim, h1_dim).normal_(0, 0.01))
        self.mu_b1 = nn.Parameter(
            torch.Tensor(h1_dim).normal_(0, 0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_b1 = nn.Parameter(torch.Tensor(h1_dim).normal_(0, 0.01))
        self.mu_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim).normal_(0, 0.01))
        self.logsigma_W2 = nn.Parameter(torch.Tensor(h1_dim, h2_dim).normal_(0, 0.01))
        self.mu_b2 = nn.Parameter(
            torch.Tensor(h2_dim).normal_(0, 0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_b2 = nn.Parameter(torch.Tensor(h2_dim).normal_(0, 0.01))
        self.mu_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension).normal_(0,
                                                                                                     0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_W3 = nn.Parameter(torch.Tensor(h2_dim, sequence_length * cw_inner_dimension).normal_(0, 0.01))
        self.mu_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length).normal_(0,
                                                                                        0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_b3 = nn.Parameter(torch.Tensor(alphabet_size * sequence_length).normal_(0, 0.01))
        self.mu_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length).normal_(0,
                                                                                                  0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_S = nn.Parameter(torch.Tensor(int(h2_dim / nb_patterns), sequence_length).normal_(0, 0.01))
        self.mu_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size).normal_(0,
                                                                                         0.01))  # try torch.randn(x,x,x)*0.01 if not working
        self.logsigma_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size).normal_(0, 0.01))
        self.mu_l = nn.Parameter(torch.Tensor(1).normal_(0, 0.01))
        self.logsigma_l = nn.Parameter(torch.Tensor(1).normal_(0, 0.01))
        self.hasTemperature = hasTemperature
        self.hasDictionary = hasDictionary

    def forward(self, z, x):
        W1 = sample_diag_gaussian_original(self.mu_W1, self.logsigma_W1)  # "W_decode_"+str(layer_num)
        b1 = sample_diag_gaussian_original(self.mu_b1, self.logsigma_b1)
        W2 = sample_diag_gaussian_original(self.mu_W2, self.logsigma_W2)
        b2 = sample_diag_gaussian_original(self.mu_b2, self.logsigma_b2)
        W3 = sample_diag_gaussian_original(self.mu_W3, self.logsigma_W3)
        b3 = sample_diag_gaussian_original(self.mu_b3, self.logsigma_b3)
        S = sample_diag_gaussian_original(self.mu_S, self.logsigma_S)
        S = S.repeat(self.nb_patterns, 1)  # W-scale
        S = F.sigmoid(S)
        if self.hasDictionary:
            W3 = W3.view(self.h2_dim * self.sequence_length, -1)
            C = sample_diag_gaussian_original(self.mu_C, self.logsigma_C)
            W_out = torch.mm(W3, C)
            S = torch.unsqueeze(S, 2)
            W_out = W_out.view(-1, self.sequence_length, self.alphabet_size)
            W_out = W_out * S
            W_out = W_out.view(-1, self.sequence_length * self.alphabet_size)
        h1 = F.relu(F.linear(z, W1.t(), b1))  # todo print h1 with deterministic z
        h2 = F.sigmoid(F.linear(h1, W2.t(), b2))
        h3 = F.linear(h2, W_out.t(), b3)
        l = sample_diag_gaussian_original(self.mu_l, self.logsigma_l)
        l = torch.log(1 + l.exp())
        h3 = h3 * l
        h3 = h3.view((-1, self.sequence_length, self.alphabet_size))
        # px_z = F.softmax(h3, 2)
        if opt.attention_decoder:
            h3_with_interaction = self.interaction(h3)
        else:
            h3_with_interaction = h3
        px_z = F.softmax(h3_with_interaction, 2)
        x = x.view(-1, self.sequence_length, self.alphabet_size)
        logpx_z = (x * F.log_softmax(h3_with_interaction, 2)).sum(-1).sum(-1)
        return px_z, logpx_z, self.mu_W1, self.logsigma_W1, self.mu_b1, self.logsigma_b1, self.mu_W2, \
                   self.logsigma_W2, self.mu_b2, self.logsigma_b2, self.mu_W3, self.logsigma_W3, self.mu_b3, \
                   self.logsigma_b3, self.mu_S, self.logsigma_S, self.mu_C, self.logsigma_C, self.mu_l, self.logsigma_l

