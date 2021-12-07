import time
import torch
import argparse
from torch.utils.data import DataLoader
from train import *
from evaluation import *
from models import *
from utils import *
from dataset import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Experiment:
    def __init__(self, config):
        self.model_name = config.get('model_name')
        self.train_conf = config.get('training')
        self.eval_desc = config.get('eval_desc')
        data_args = {
            'neg_ratio': self.train_conf.get('neg_ratio'),
            'batch_sample': self.train_conf.get('batch_sample')
        }
        self.dataset = Dataset(config.get('dataset'), data_args)
        config['entity_cnt'] = len(self.dataset.data['entity'])
        config['relation_cnt'] = len(self.dataset.data['relation'])
        self.model, self.device = init_model(config)
        if self.model_name in ['ConvE', 'ConvR']:
            self.train_func = train_without_label
            self.eval_func = eval_for_tail
            self.output_func = output_eval_tail
        elif self.model_name in ['ConvKB', 'TransE']:
            self.train_func = train_with_label
            self.eval_func = eval_for_both
            self.output_func = output_eval_both
        else:
            logging.error(f'Could not find any training function for model={self.model_name}')
        opt_conf = config.get('optimizer')
        if opt_conf.get('algorithm') == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        elif opt_conf.get('algorithm') == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        elif opt_conf.get('algorithm') == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        else:
            logging.error('Could not find corresponding optimizer for algorithm={}'.format(opt_conf.get('algorithm')))
        self.do_validation = config.get('do_validation')
        self.valid_steps = config.get('valid_steps')
        self.do_test = config.get('do_test')
        self.save_model_path = config.get('save_model_path')
    
    def train_and_eval(self):
        train_loader = DataLoader(self.dataset.data['train'], self.train_conf.get('batch_size'), shuffle=self.train_conf.get("shuffle"), drop_last=False)
        if self.dataset.data['valid']:
            valid_loader = DataLoader(self.dataset.data['valid'], self.train_conf.get('batch_size'), shuffle=False, drop_last=False)
        if self.dataset.data['test']:
            test_loader = DataLoader(self.dataset.data['test'], self.train_conf.get('batch_size'), shuffle=False, drop_last=False)
        for epoch in range(self.train_conf.get('epochs')):
            logging.info('Start training epoch: %d' % (epoch + 1))
            start_time = time.time()
            epoch_loss = self.train_func(train_loader, self.model, self.optimizer, self.device)
            end_time = time.time()
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            print('[Epoch #%d] training loss: %.6f - training time: %.2f seconds' % (epoch + 1, mean_loss, end_time - start_time))
            if self.do_validation and (epoch + 1) % self.valid_steps == 0:
                print(f'--- epoch #{epoch + 1} valid ---')
                logging.info('Start evaluation of validation data')
                self.model.eval()
                with torch.no_grad():
                    eval_results = self.eval_func(valid_loader, self.model, self.device, self.dataset.data, self.eval_desc)
                    self.output_func(eval_results, 'validation')
        if self.do_test:
            print(f'--- test ---')
            logging.info('Start evaluation on test data')
            self.model.eval()
            with torch.no_grad():
                eval_results = self.eval_func(test_loader, self.model, self.device, self.dataset.data, self.eval_desc)
                self.output_func(eval_results, 'test')
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
            logging.info('Created output directory {}'.format(self.save_model_path))
        torch.save(self.model, f'{self.save_model_path}/{self.model_name}_{self.dataset.name}.ckpt')
        logging.info('Finished! Model saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge graph inference arguments.')
    parser.add_argument('-c', '--config', dest='config_file', help='The path of configuration json file.')
    args = parser.parse_args()

    config = load_json_config(args.config_file)
    print(config)

    experiment = Experiment(config)
    experiment.train_and_eval()
