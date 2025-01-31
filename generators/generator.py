# here put the import lib
import os
import time
import pickle
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from generators.data import SeqDataset
from generators.data import CDSRSeq2SeqDataset, CDSREvalSeq2SeqDataset
from generators.data import CDSRRegSeq2SeqDatasetUser
from utils.utils import unzip_data, concat_data, normalize, sparse_mx_to_torch_sparse_tensor


class Generator(object):

    def __init__(self, args, logger, device):

        self.args = args
        self.aug_file = args.aug_file
        self.inter_file = args.inter_file
        self.dataset = args.dataset
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device
        self.aug_seq = args.aug_seq

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    
    def _load_dataset(self):
        '''Load train, validation, test dataset'''

        user_train, user_valid, user_test = {}, {}, {}
        domain_train, domain_valid, domain_test = {}, {}, {}
        # assume user/item index starting from 1
        inter_seq, domain_seq = pickle.load(open('./data/%s/handled/%s.pkl' % (self.dataset, self.inter_file), 'rb'))
        id_map = json.load(open("./data/{}/handled/id_map.json".format(self.dataset)))

        self.user_num = max(id_map["user_dict"]["str2id"].values())
        self.item_num_dict = id_map["item_dict"]["item_count"]
        self.item_num = self.item_num_dict["0"] + self.item_num_dict["1"]

        for user in tqdm(inter_seq.keys()):
            nfeedback = len(inter_seq[user])
            #nfeedback = len(User[user])
            if nfeedback < 3:
                continue
                user_train[user] = inter_seq[user]
                user_valid[user] = []
                user_test[user] = []
                domain_train[user] = domain_seq[user]
                domain_valid[user] = []
                domain_test[user] = []
            else:
                user_train[user] = inter_seq[user][:-2]
                user_valid[user] = []
                user_valid[user].append(inter_seq[user][-2])
                user_test[user] = []
                user_test[user].append(inter_seq[user][-1])
                domain_train[user] = domain_seq[user][:-2]
                domain_valid[user] = []
                domain_valid[user].append(domain_seq[user][-2])
                domain_test[user] = []
                domain_test[user].append(domain_seq[user][-1])
        
        self.train, self.valid, self.test = user_train, user_valid, user_test
        self.domain_train, self.domain_valid, self.domain_test = domain_train, domain_valid, domain_test


    
    def make_trainloader(self):

        train_dataset = unzip_data(self.train)
        self.train_dataset = SeqDataset(train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
    

        return train_dataloader


    def make_evalloader(self, test=False):

        if test:
            eval_dataset = concat_data([self.train, self.valid, self.test])

        else:
            eval_dataset = concat_data([self.train, self.valid])

        self.eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg)
        eval_dataloader = DataLoader(self.eval_dataset,
                                    sampler=SequentialSampler(self.eval_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return eval_dataloader

    
    def get_user_item_num(self):

        return self.user_num, self.item_num
    

    def get_item_num_dict(self):

        return self.item_num_dict
    

    def get_item_pop(self):
        """get item popularity according to item index. return a np-array"""
        all_data = concat_data([self.train, self.valid, self.test])
        pop = np.zeros(self.item_num+1) # item index starts from 0
        
        for items in all_data:
            pop[items] += 1

        return pop
    

    def get_user_len(self):
        """get sequence length according to user index. return a np-array"""
        all_data = concat_data([self.train, self.valid])
        lens = []

        for user in all_data:
            lens.append(len(user))

        return np.array(lens)
    

    def load_adj(self):

        adj, adj_A, adj_B = pickle.load(open('./data/%s/handled/adj.pkl' % self.dataset, 'rb'))
        adj = sparse_mx_to_torch_sparse_tensor(normalize(adj))
        adj_A = sparse_mx_to_torch_sparse_tensor(normalize(adj_A))
        adj_B = sparse_mx_to_torch_sparse_tensor(normalize(adj_B))

        return adj, adj_A, adj_B
    

    def load_adj_attr(self):

        adj = pickle.load(open('./data/%s/handled/adj_attr.pkl' % self.dataset, 'rb'))
        node_num = adj.shape[0]
        adj = sparse_mx_to_torch_sparse_tensor(normalize(adj))

        return adj, node_num
    


class CDSRSeq2SeqGenerator(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):

        train_dataset = unzip_data(self.train)
        train_domain = unzip_data(self.domain_train)
        self.train_dataset = CDSRSeq2SeqDataset(self.args, 
                                                train_dataset, 
                                                train_domain,
                                                self.item_num_dict, 
                                                self.args.max_len, 
                                                self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader
    

    def make_evalloader(self, test=False):

        if test:
            eval_dataset = concat_data([self.train, self.valid, self.test])
            eval_domain = concat_data([self.domain_train, self.domain_valid, self.domain_test])
        else:
            eval_dataset = concat_data([self.train, self.valid])
            eval_domain = concat_data([self.domain_train, self.domain_valid])

        self.eval_dataset = CDSREvalSeq2SeqDataset(self.args,
                                                   eval_dataset, 
                                                   eval_domain,
                                                   self.item_num_dict, 
                                                   self.args.max_len, 
                                                   self.args.test_neg)
        
        eval_dataloader = DataLoader(self.eval_dataset,
                                    sampler=SequentialSampler(self.eval_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return eval_dataloader


    def _load_cold_dataset(self):
        '''Load cold-start dataset for test'''

        user_cold = {}
        domain_cold = {}
        # assume user/item index starting from 1
        inter_seq, domain_seq = pickle.load(open('./data/%s/handled/%s_cold.pkl' % (self.dataset, self.inter_file), 'rb'))

        for user in tqdm(inter_seq.keys()):
            nfeedback = len(inter_seq[user])
            #nfeedback = len(User[user])
            if nfeedback < 3:
                continue
            else:
                user_cold[user] = inter_seq[user]
                domain_cold[user] = domain_seq[user]
        
        return user_cold, domain_cold


    
    def make_coldloader(self):

        cold_dataset, cold_domain = self._load_cold_dataset()
        cold_dataset = unzip_data(cold_dataset)
        cold_domain = unzip_data(cold_domain)

        self.cold_dataset = CDSREvalSeq2SeqDataset(self.args,
                                                   cold_dataset, 
                                                   cold_domain,
                                                   self.item_num_dict, 
                                                   self.args.max_len, 
                                                   self.args.test_neg)
        cold_dataloader = DataLoader(self.cold_dataset,
                                    sampler=SequentialSampler(self.cold_dataset),
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return cold_dataloader

    


class CDSRRegSeq2SeqGeneratorUser(CDSRSeq2SeqGenerator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):

        train_dataset = unzip_data(self.train)
        train_domain = unzip_data(self.domain_train)
        self.train_dataset = CDSRRegSeq2SeqDatasetUser(
                                                        self.args, 
                                                        train_dataset, 
                                                        train_domain,
                                                        self.item_num_dict, 
                                                        self.args.max_len, 
                                                        self.args.train_neg
                                                        )

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader




