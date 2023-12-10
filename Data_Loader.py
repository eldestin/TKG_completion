import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from collections import defaultdict

# Here is the Implementation dataset reading
class Data:
    def __init__(self, dataset = None, add_reverse_relation = False, path = None):
        '''
        Params:
            1. Dataset: Chosen dataset;
            2. add_reverse_relation: whether to add reverse part
            3. path: read in dataset path;
        '''
        self.dataset = dataset
        self.add_reverse_relation = add_reverse_relation
        self.path = path
        self.ent2id = self.read_id2entity()
        self.rel2id = self.read_id2relation()
        num_rels = len(self.rel2id)
        self.init_rels = len(self.rel2id)
        reverse = {}
        self.num_rels = len(self.rel2id)
        if add_reverse_relation:
            for ind, rel in self.rel2id.items():
                reverse[ind + len(self.rel2id)] = "Reverse_" + rel
            self.rel2id.update(reverse)
            self.num_rels = 2 * num_rels
        else:
            self.num_rels = num_rels

        self.num_ents = len(self.ent2id)
        self.rel2id[self.num_rels] = "selfloop"
        
        self.train_set = self.load_data(mode = "train")
        self.valid_set = self.load_data(mode = "valid")
        self.test_set = self.load_data(mode = "test")
        
        # add reverse to dataset
        if self.add_reverse_relation:
            self.train_set = self.add_reverse(self.train_set)
        self.seen_entities = set(self.train_set[:, 0]).union(self.train_set[:, 2])
        self.seen_relations = set(self.train_set[:,1])
        self.val_mask = self.create_mask(self.valid_set)
        self.seen_val = self.filtering(self.valid_set)
        
        if self.add_reverse_relation:
            self.valid_set = self.add_reverse(self.valid_set)
            self.valid_data_seen_entity = np.concatenate([self.seen_val[:, :-1],
                                                         np.vstack([[triple[2], triple[1] + num_rels, triple[0], triple[3]]
                                                                        for triple in self.valid_set])], axis = 0)
        
        test_mask = self.create_mask(self.test_set)
        test_mask_conjugate = ~np.array(test_mask)
        
        print('Seen data proportion: ' + str(np.asarray(test_mask).sum()/len(test_mask)))
        print('Unseen data proportion: '+ str(test_mask_conjugate.sum()/test_mask_conjugate.size))
        self.seen_test = self.test_set[test_mask]
        self.unseen_test = self.test_set[test_mask_conjugate]
        if self.add_reverse_relation:
            self.test_set = self.add_reverse(self.test_set)
            self.seen_test = self.add_reverse(self.seen_test)
            self.unseen_test = self.add_reverse(self.unseen_test)
        # dataset statistic
        print("Number of entities: {}".format(self.num_ents))
        print("Number of relations: {}".format(self.num_rels//2)) if self.add_reverse_relation else print("Number of relations: {}".format(self.num_rels))
        print("length of training: ", len(self.train_set))
        print("length of validation: ", len(self.valid_set))
        print("length of test: ", len(self.test_set))
        self.data_all = np.concatenate([self.train_set, self.valid_set, self.test_set], axis = 0)
        self.timestamps = self.get_timestamp(self.data_all)
        self.total_len = len(self.data_all)
        
        # create label: true object with correct time info
        # if without time
        self.spt2o = defaultdict(list)
        for triple in self.data_all:
            self.spt2o[(triple[0], triple[1], triple[-1])].append(triple[2])
        # final checking and filtering
        self.train_set = self.prepare_input(self.train_set)
        self.train_label = self.get_label(self.train_set)
        self.valid_set = self.prepare_input(self.valid_set)
        self.valid_label = self.get_label(self.valid_set)
        self.test_set = self.prepare_input(self.test_set)
        self.test_label = self.get_label(self.test_set)
        self.get_edge_idx()
        print("Train, valid, test set proportion: {:.3f} {:.3f} {:.3f}".format(len(self.train_set)/self.total_len,len(self.valid_set)/self.total_len,len(self.test_set)/self.total_len ))
        print("Complete loading data!")
    def get_label(self, data):
        '''
        Given data: get the label as: (s,r,t) -> o
        '''
        labels = []
        for triple in data:
            labels.append(self.spt2o[(triple[0], triple[1], triple[-1])])
        return labels 
    def read_id2entity(self):
        with open(os.path.join(self.path, self.dataset, "entity2id.txt"), 'r', encoding = 'utf-8') as f:
            mapping = f.readlines()
            mapping = [entity.strip().split("\t") for entity in mapping]
            mapping = {int(ent2idx[1].strip()): ent2idx[0].strip() for ent2idx in mapping}
        return mapping
    def read_id2relation(self):
        with open(os.path.join(self.path, self.dataset, "relation2id.txt"), 'r', encoding = 'utf-8') as f:
            mapping = f.readlines()
            mapping = [relation.strip().split("\t") for relation in mapping]
            mapping = {int(rel2idx[1].strip()): rel2idx[0].strip() for rel2idx in mapping}
        return mapping
    def load_data(self, mode = "train"):
        with open(os.path.join(self.path, self.dataset, "{}.txt".format(mode)), 'r', encoding = 'utf-8') as f:
            triples = f.readlines()
            triples = np.array([line.split("\t") for line in triples])
            triples = np.vstack([[int(line.strip()) for line in lines] for lines in triples])
        return triples
    def add_reverse(self, dataset):
        '''
        Params:
            Add reverse to dataset;
            dataset: a numpy array (2D), num_triples * (s,r,o,t)
        '''
        #print(dataset[:,-1])
        r_dataset = np.concatenate([dataset[:, :-1], 
                                   np.vstack([[triple[2], triple[1] + self.init_rels, triple[0], triple[3]] for triple in dataset])], axis = 0)
        return r_dataset
    def create_mask(self, dataset):
        '''
        Create mask for seen triples in KG
        '''
        mask = [triple[0] in self.seen_entities and triple[1] in self.seen_relations and triple[2] in self.seen_entities for triple in dataset]
        return mask
    def filtering(self, dataset, conj = False):
        return dataset[self.create_mask(dataset)] if not conj else dataset[~np.array(self.create_mask(dataset))]
    
    def get_timestamp(self, dataset):
        timestamps = np.array(sorted(list(set(d[3] for d in dataset))))
        return timestamps
    
    def get_adj_dict(self):
        adj_dict = defaultdict(list)
        for triple in self.data_all:
            adj_dict[int(triple[0])].append((int(triple[2]), int(triple[1]), int(triple[3])))
        for value in adj_dict.values():
            # sort by time stamp
            value.sort(key = lambda x: (x[2], x[0], x[1]))
        return adj_dict

    def prepare_input(self, dataset, start_time = 0):
        # Spliting strategy
        # Here split the data into train, valid and test set, assert that train time stamp < valid time stamp < test time stamp
        '''
        This method check the data again and remove the triples without time stamp (-1)
        Params: 
            1. dataset: data array;
            2. mode: train, valid or test;
            3. start time: start filtering time;
        '''
        assert max(dataset[:, 3]) >= start_time, "start time should be smaller than filtering max time."
        triples = np.vstack([np.array(triple) for triple in dataset if triple[3] >= start_time]) 
        return triples
 
        
    def get_spt2o(self, mode):
        if mode == "train":
            triples = self.train_set
        elif mode == "valid":
            triples = self.valid_set
        elif mode == "test":
            triples = self.test_set
        else:
            raise ValueError("Invalid input mode. ")
        spt2o = defaultdict(list)
        for triple in triples:
            spt2o[(triple[0], triple[1], triple[-1])].append(triple[2])
        return spt2o
    
    def get_edge_idx(self):
        '''
        This function return the graph into edge index and edge type format:
            1. edge idx: [2, F]: [h, t] for each triple, F: number of facts;
            2. edge type: [1, F]: [r], for each triple;
        '''
        self.train_edge_idx = [self.train_set[:,0].tolist(), self.train_set[:,2].tolist()]
        self.valid_edge_idx = [self.valid_set[:,0].tolist(), self.valid_set[:,2].tolist()]
        self.test_edge_idx = [self.test_set[:,0].tolist(), self.test_set[:,2].tolist()]
        
        self.train_edge_type = self.train_set[:, 1].tolist()
        self.valid_edge_type = self.valid_set[:, 1].tolist()

        self.test_edge_type = self.test_set[:, 1].tolist()


# create dataset and data loader
class KG_dataset(Dataset):
    def __init__(self, dataset, num_entities, num_rels, labels, neg_sampling = False):
        '''
        Torch dataset:
            Params: 
                dataset: the numpy ndarray;
                labels: corresponding ground truth label
        '''
        super(KG_dataset,self).__init__()
        self.num_entities = num_entities
        self.num_rels = num_rels
        self.dataset = dataset
        # label list
        self.labels = labels
        # source index (h, r, t) -> h
        self.src_idx = self.dataset[:, 0].tolist()
        self.rel_idx = self.dataset[:, 1].tolist()
        self.target_idx = self.dataset[:, 2].tolist()
        self.ts = self.dataset[:, -1].tolist()
        self.neg_sampling = neg_sampling
        if self.neg_sampling:
            self.neg_data = self.generate_neg()
            self.neg_src_idx = self.neg_data[:, 0].tolist()
            self.neg_rel_idx = self.neg_data[:, 1].tolist()
            self.neg_target_idx = self.neg_data[:, 2].tolist()
            self.neg_ts = self.neg_data[:, -1].tolist()
            print(len(self.dataset), len(self.neg_data))
    def generate_neg(self):
        import random
        neg_candidates, i = [],0
        neg_data = []
        candidates = list(range(self.num_entities))
        for idx, triple in enumerate(self.dataset):
            while True:
                if i == len(neg_candidates):
                    i = 0 
                    neg_candidates = random.choices(population = candidates, k = int(1e4))
                neg, i = neg_candidates[i], i+1
                # 随机替换头，尾
                if random.randint(0,1) == 1:
                    neg_data.append([neg, triple[1], triple[2], triple[-1]])
                    break
                else:
                    neg_data.append([triple[0], triple[1], neg, triple[-1]])
                    break
        return np.array(neg_data)
    
    def get_label(self, label):
        '''
        Let index to 1.0, else 0
        '''
        y = np.zeros([self.num_entities], dtype = np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # process label data
        batch_label = self.labels[idx]
        batch_label = self.get_label(batch_label)
        if not self.neg_sampling:
            return (self.src_idx[idx], self.rel_idx[idx], self.target_idx[idx], self.ts[idx]), batch_label
        else: 
            return (self.src_idx[idx], self.rel_idx[idx], self.target_idx[idx], self.ts[idx]) ,(self.neg_src_idx[idx], self.neg_rel_idx[idx], self.neg_target_idx[idx], self.neg_ts[idx]), batch_label