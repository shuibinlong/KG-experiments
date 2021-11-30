import os
import random
import logging
import numpy as np
from tqdm import tqdm

from utils import load_triples, load_ids

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Dataset:
    def __init__(self, dataset, kwargs):
        self.path = os.getcwd()
        self.name = dataset
        self.data = {
            'train': self.read_train(),
            'valid': self.read_valid(),
            'test': self.read_test(),
            'entity': self.read_entity(),
            'relation': self.read_relation(),
            'entity_relation': {}
        }

        self.gen_entity_relation_multidata()

        if kwargs.get('neg_ratio') > 0:
            self.data['train'] = self.neg_sampling(kwargs.get('neg_ratio'), kwargs.get('batch_sample'))
    
    def read_train(self):
        logging.info(' Loading training data '.center(100, '-'))
        return load_triples(os.path.join(self.path, 'data', self.name, 'train2id.txt'))
    
    def read_valid(self):
        logging.info(' Loading validation data '.center(100, '-'))
        return load_triples(os.path.join(self.path, 'data', self.name, 'valid2id.txt'))

    def read_test(self):
        logging.info(' Loading testing data '.center(100, '-'))
        return load_triples(os.path.join(self.path, 'data', self.name, 'test2id.txt'))
    
    def read_entity(self):
        logging.info(' Loading entity id '.center(100, '-'))
        return load_ids(os.path.join(self.path, 'data', self.name, 'entity2id.txt'))
    
    def read_relation(self):
        logging.info(' Loading realtion id '.center(100, '-'))
        return load_ids(os.path.join(self.path, 'data', self.name, 'relation2id.txt'))
    
    def gen_entity_relation_multidata(self):
        logging.info(' Generating entity-relation dictionaries to accelerate evaluation process '.center(100, '-'))
        full_data = self.data['train'] + self.data['valid'] + self.data['test']
        self.data['entity_relation']['as_head'] = {}
        self.data['entity_relation']['as_tail'] = {}
        for i in self.data['entity']:
            self.data['entity_relation']['as_head'][i] = {}
            self.data['entity_relation']['as_tail'][i] = {}
            for j in self.data['relation']:
                self.data['entity_relation']['as_head'][i][j] = []
                self.data['entity_relation']['as_tail'][i][j] = []
        for triple in full_data:
            h = triple[0]
            t = triple[1]
            r = triple[2]
            self.data['entity_relation']['as_head'][t][r].append(h)
            self.data['entity_relation']['as_tail'][h][r].append(t)

    def neg_sampling(self, neg_ratio, batching):
        logging.info(' Sampling corrupted triples '.center(100, '-'))
        train_data = []
        entity_set = set(self.data['entity'])
        for triple in tqdm(self.data['train'], total=len(self.data['train'])):
            train_data.append([*triple, 1]) # positive
            h, t, r = triple
            if batching:
                h_neg_sampling = random.sample(entity_set - set(self.data['entity_relation']['as_head'][t][r]), (neg_ratio + 1) // 2)
                t_neg_sampling = random.sample(entity_set - set(self.data['entity_relation']['as_tail'][h][r]), (neg_ratio + 1) // 2)
            else:
                h_neg_sampling = []
                t_neg_sampling = []
                for _ in range(neg_ratio):
                    if np.random.binomial(1, 0.5):
                        h_neg = random.sample(entity_set - set(self.data['entity_relation']['as_head'][t][r]), 1)[0]
                        h_neg_sampling.append(h_neg)
                    else:
                        t_neg = random.sample(entity_set - set(self.data['entity_relation']['as_tail'][h][r]), 1)[0]
                        t_neg_sampling.append(t_neg)
            for h_neg in h_neg_sampling:
                train_data.append([h_neg, t, r, -1])
            for t_neg in t_neg_sampling:
                train_data.append([h, t_neg, r, -1])
        return train_data
