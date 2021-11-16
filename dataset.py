import os
import random
import logging
import numpy as np

from utils import load_triples, load_ids

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Dataset:
    def __init__(self, dataset, neg_ratio):
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

        if neg_ratio > 0:
            self.data['train'] = self.neg_sampling(neg_ratio)
    
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
            self.data['entity_relation']['as_head'][h][r].append(t)
            self.data['entity_relation']['as_tail'][t][r].append(h)

    def neg_sampling(self, neg_ratio):
        logging.info(' Sampling corrupted triples '.center(100, '-'))
        train_data = []
        entity_set = set(self.data['entity'])
        entity_relation_as_head = set(self.data['entity_relation']['as_head'])
        entity_relation_as_tail = set(self.data['entity_relation']['as_tail'])
        for triple in self.data['train']:
            train_data.append([*triple, 1]) # positive
            h_neg = triple[0]
            t_neg = triple[1]
            r = triple[2]
            used = {'head': set(), 'tail': set()}
            for _ in range(neg_ratio):
                # 1/2 probability to replace the head entity
                if np.random.binomial(1, 0.5): # sample a negative head entity
                    h_neg = random.sample(entity_set - entity_relation_as_head[t_neg][r] - used['head'], 1)[0]
                    used['head'].add(h_neg)
                else:  # sample a negative tail entity
                    t_neg = random.sample(entity_set - entity_relation_as_tail[h_neg][r] - used['tail'], 1)[0]
                    used['tail'].add(t_neg)
                train_data.append([h_neg, t_neg, r, 0]) # negative
        return train_data
