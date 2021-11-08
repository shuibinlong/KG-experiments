import os
import logging
from utils import load_triples, load_ids

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Dataset:
    def __init__(self, dataset):
        self.path = os.getcwd()
        self.name = dataset
        self.data = {
            'train': self.read_train(),
            'valid': self.read_valid(),
            'test': self.read_test(),
            'entity': self.read_entity(),
            'relation': self.read_relation()
        }
    
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
