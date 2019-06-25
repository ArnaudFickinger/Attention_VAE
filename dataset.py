###
'''
April 2019
Code by: Arnaud Fickinger
'''
###

import torch
from utils import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_helper):
        super(Dataset, self).__init__()

        all_data = dataset_helper.x_train
        self.data = all_data
        # total_len = all_data.shape[0]
        # if isTrain:
        #     self.len = int(0.8 * total_len)
        #     self.data = all_data[:self.len,:,:]
        # else:
        #     self.len = total_len - int(0.8 * total_len)
        #     self.data = all_data[int(0.8 * total_len):,:,:]


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
