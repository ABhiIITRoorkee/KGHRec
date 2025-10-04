import os
import time
import random
import collections

import torch
import numpy as np
import pandas as pd


class DataLoaderBase(object):

    """
    Base data loader class for collaborative filtering and knowledge graph data.
    Used by higher-level loaders such as DataLoaderKGHRec.
    """
    
    def __init__(self, args, logging):
        """
        Initialize data paths, load datasets, and prepare statistics.
        """
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        self.data_dir = os.path.join(args.data_dir, args.data_name, args.path)
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.kg_file = os.path.join(self.data_dir, "relations.txt")
        self.kg_empty = os.path.join(self.data_dir, "relation_list.txt")
        
        #determins if use attention mechanism in the model
        self.attention = args.attention
        
        #determins if use knowledge graph in the model
        self.knowledgegraph = args.knowledgegraph

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
                                                #load_cf is self-defined function as shown in Line 33
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
        self.statistic_cf()

        if self.use_pretrain == 1:
            self.load_pretrained_data()

    # -----------------------------------------------------------
    #  Collaborative Filtering (CF) Data
    # -----------------------------------------------------------


    def load_cf(self, filename):
        """
        Load collaborative filtering (user-item/project-library) data from file.
        """
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict


    def statistic_cf(self):
        """
        Compute and log dataset statistics for users/projects and items/libraries.
        """

        #max(self.cf_train_data[0]) is the maximum ID of current projects/users
        #here, max(self.cf_train_data[0])-min(self.cf_train_data[0]) is the real user/project amount, as the user/project IDs do not start from 0 (offset applied).
        print(" - - - - - test - - - - - -")
        print("cf_train_data max user(project) id is " + str(max(self.cf_train_data[0])))
        print("cf_train_data min user(project) id is " + str(min(self.cf_train_data[0])))
        print("cf_test_data max user(project) id is " + str(max(self.cf_test_data[0])))
        print("cf_test_data min user(project) id is " + str(min(self.cf_test_data[0])))
        print(" - - - - - test - - - - - -")
        self.n_users = max(max(self.cf_train_data[0])-min(self.cf_train_data[0]), max(self.cf_test_data[0])-min(self.cf_test_data[0])) + 1
        
        print(" - - - - - test - - - - - -")
        print("cf_train_data max item(TPL) id is " + str(max(self.cf_train_data[0])))
        print("cf_train_data min item(TPL) id is " + str(min(self.cf_train_data[0])))
        print("cf_test_data max item(TPL) id is " + str(max(self.cf_test_data[0])))
        print("cf_test_data min item(TPL) id is " + str(min(self.cf_test_data[0])))
        print(" - - - - - test - - - - - -")        
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        
    # -----------------------------------------------------------
    #  Knowledge Graph (KG) Data
    # -----------------------------------------------------------

    def load_kg(self, filename):

        """
        Load knowledge graph data of (head, relation, tail) triples.
        """
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data


    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        """
        Sample positive items for a given user.
        """

        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items


    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        """
        Sample negative items (not interacted) for a given user.
        """
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items


    def generate_cf_batch(self, user_dict, batch_size):
        """
        Generate a mini-batch for CF training.
        Returns (user, positive_item, negative_item).
        """

        exist_users = user_dict.keys()
        if batch_size <= len(exist_users):
            #batch_user = random.sample(exist_users, batch_size)
            batch_user = random.sample(sorted(exist_users), batch_size)


        else:
            #batch_user = [random.choice(exist_users) for _ in range(batch_size)]
            batch_head = [random.choice(list(exist_users)) for _ in range(batch_size)]


        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        """
        Sample positive (relation, tail) pairs for a given head entity.
        """
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        """
        Sample negative tail entities for a given (head, relation) pair.
        """
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        """
        Generate a mini-batch for KG training.
        Returns (head, relation, pos_tail, neg_tail).
        """
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            #batch_head = random.sample(exist_heads, batch_size)
            batch_head = random.sample(sorted(exist_heads), batch_size)


        else:
            #batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
            batch_head = [random.choice(list(exist_heads)) for _ in range(batch_size)]

            

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    # -----------------------------------------------------------
    #  Pretrained Embeddings
    # -----------------------------------------------------------

    def load_pretrained_data(self):
        """
        Load pretrained user/item embeddings for initialization.
        """
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users, "User embedding shape mismatch"
        assert self.item_pre_embed.shape[0] == self.n_items, "Item embedding shape mismatch"
        assert self.user_pre_embed.shape[1] == self.args.embed_dim, "User embedding dim mismatch"
        assert self.item_pre_embed.shape[1] == self.args.embed_dim, "Item embedding dim mismatch"

