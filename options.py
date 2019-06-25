###
'''
April 2019
Code by: Arnaud Fickinger
'''
###

import argparse
import os


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        #Attention Parameters
        self.parser.add_argument('--query_dim', type=int, default=512)
        self.parser.add_argument('--interaction_inner_dim', type=int, default=2048)
        self.parser.add_argument('--avae', dest='avae', action='store_true', default=False)
        self.parser.add_argument('--attention_encoder', action='store_true', default=True)
        self.parser.add_argument('--attention_decoder', action='store_true', default=True)
        self.parser.add_argument('--dropout_att', type=float, default=0.1)

        #Training parameter
        self.parser.add_argument('--is_train', dest='is_train', action='store_true', default=False)

        #Pytorch parameters
        self.parser.add_argument('--random_seed', type=int, default=42)
        self.parser.add_argument('--warm_up', type=int, default=0)
        self.parser.add_argument('--saving_path', type=str, default="./checkpoint/")
        self.parser.add_argument('--save_model_every', type=int, default=30000)
        self.parser.add_argument('--save_plot_every', type=int, default=30000)
        self.parser.add_argument('--save_spearman_every', type=int, default=30000)
        self.parser.add_argument('--continue_training', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=100)
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--neff', type=float, default = 0)#535.5874
        self.parser.add_argument('--theano_test', dest='theano_test', action='store_true', default=False)
        self.parser.add_argument('--dropout', type=float, default=0)


        #Parameters of the data
#         self.parser.add_argument('--dataset', type=str, default="DLG4_RAT")#'BLAT_ECOLX'
#         self.parser.add_argument('--theta', type=float, default=0.2)
        self.parser.add_argument('--alphabet_size', type=int, default=20, help='size of the alphabet')
        self.parser.add_argument('--sequence_length', type=int, default=84, help='length of the sequence') #253

        #Parameters of the VAE
        self.parser.add_argument('--latent_dim', type=int, default=30, help='dimension of the latent space')
        self.parser.add_argument('--is_sparse', action='store_true', default=True)
        self.parser.add_argument('--has_temperature', action='store_true', default=True)
        self.parser.add_argument('--has_dictionary', action='store_true', default=True)
        self.parser.add_argument('--is_semi_supervised', action='store_true', default=False)


        #Parameters of the encoder
        self.parser.add_argument('--enc_h1_dim', type=int, default=1500, help='dimension of the first hl of the enc')
        self.parser.add_argument('--enc_h2_dim', type=int, default=1500, help='dimension of the second hl of the enc')

        #Parameters of the decoder
        self.parser.add_argument('--dec_h1_dim', type=int, default=100, help='dimension of the first hl of the dec')
        self.parser.add_argument('--dec_h2_dim', type=int, default=2000, help='dimension of the second hl of the dec')
        self.parser.add_argument('--cw_inner_dimension', type=int, default=40, help='inner dimension of the product CW')

        #Sparsity parameters
        self.parser.add_argument('--nb_patterns', type=int, default=4, help='nb of times S is repeated')
        self.parser.add_argument('--mu_sparse', type=float, default=-12.36, help='sparse prior mean')
        self.parser.add_argument('--logsigma_sparse', type=float, default=0.602, help='sparse prior sigma')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt