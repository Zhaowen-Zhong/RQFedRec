import random
import numpy as np
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, dataset_dict):
        self.set_dict(dataset_dict)
        self.trainset = np.load(self.path + '/train_dataset.npy')
        self.validset = np.load(self.path + '/valid_dataset.npy')
        self.testset = np.load(self.path + '/test_dataset.npy')
        self.sp_trainset = sp.csr_matrix(self.trainset)
        self.sp_validset = sp.csr_matrix(self.validset)
        self.sp_testset = sp.csr_matrix(self.testset)
        self.user_num = self.trainset.shape[0]
        self.item_num = self.trainset.shape[1]
        self.users = list(range(self.user_num))
        self.getlabel()
        self.sample()

    def set_dict(self, dataset_dict):
        self.path = dataset_dict['path']
        self.sample_method = dataset_dict['sample_method']
        self.num_negative = dataset_dict.getint('num_negative')

    def getlabel(self):
        self.validlabel = list()
        for valid_line in self.validset:
            indices = [i for i, x in enumerate(valid_line) if x > 0]
            self.validlabel.append(indices)
        self.testlabel = list()
        for test_line in self.testset:
            indices = [i for i, x in enumerate(test_line) if x > 0]
            self.testlabel.append(indices)

    def getUserPosItems(self):
        self.trainposItems = dict()
        self.validposItems = dict()
        self.testposItems = dict()
        for user in self.users:
            self.trainposItems[user] = self.sp_trainset[user].nonzero()[1]
            self.validposItems[user] = self.sp_validset[user].nonzero()[1]
            self.testposItems[user] = self.sp_testset[user].nonzero()[1]

    def sample(self):
        self.getUserPosItems()
        self.UniformSample()
        if self.sample_method == 'all':
            self.valid = torch.tensor(range(self.item_num)).repeat(self.user_num, 1)
            self.test = torch.tensor(range(self.item_num)).repeat(self.user_num, 1)
        elif self.sample_method == 'single':
            self.ValidSample()
        else:
            raise NotImplementedError

    def UniformSample(self):
        S = list()
        all_item_num = np.arange(self.item_num)
        for user in self.users:
            posForUser = self.trainposItems[user]
            negforUser_all = np.setdiff1d(all_item_num, posForUser)
            for pos in posForUser:
                S.append([user, pos, 1.0])
                negitem_list = random.sample(list(negforUser_all), self.num_negative)
                for negitem in negitem_list:
                    S.append([user, negitem, 0.0])
        d1, d2, d3 = zip(*S)
        self.S = [torch.tensor(d1).to(torch.long), torch.tensor(d2).to(torch.long), torch.tensor(d3)]

    def ValidSample(self):
        self.valid = list()
        self.test = list()
        all_item_num = np.arange(self.item_num)
        for user in self.users:
            posForUser = np.concatenate((self.trainposItems[user], self.validposItems[user], self.testposItems[user]))
            negforUser_all = np.setdiff1d(all_item_num, posForUser)
            negitem_list = random.sample(list(negforUser_all), 99)
            self.valid.append(self.validposItems[user].tolist() + negitem_list)
            self.test.append(self.testposItems[user].tolist() + negitem_list)

    def __len__(self):
        return len(self.S[0])

    def __getitem__(self, index):
        return self.S[0][index], self.S[1][index], self.S[2][index]
