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
        self.dataset = Dataset(config.get('dataset'))
        config['entity_cnt'] = len(self.dataset.data['entity'])
        config['relation_cnt'] = len(self.dataset.data['relation'])
        self.model, self.device = init_model(config)
        self.train_conf = config.get('training')
        opt_conf = config.get('optimizer')
        if opt_conf.get('algorithm', 'adam') == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt_conf.get('lr'), weight_decay=opt_conf.get('weight_decay'))
        self.do_validation = config.get('do_validation')
        self.valid_steps = config.get('valid_steps')
        self.do_test = config.get('do_test')
        self.save_model_path = config.get('save_model_path')
    
    def train_and_eval(self):
        train_loader = DataLoader(self.dataset.data['train'], self.train_conf.get('batch_size'), shuffle=True, drop_last=False)
        if self.dataset.data['valid']:
            valid_loader = DataLoader(self.dataset.data['valid'], self.train_conf.get('batch_size'), shuffle=False, drop_last=False)
        if self.dataset.data['test']:
            test_loader = DataLoader(self.dataset.data['test'], self.train_conf.get('batch_size'), shuffle=False, drop_last=False)
        for epoch in range(self.train_conf.get('epochs')):
            logging.info('Start training epoch: %d' % (epoch + 1))
            start_time = time.time()
            if self.model_name in ['ConvE', 'ConvR']:
                epoch_loss = train_conv(train_loader, self.model, self.optimizer, self.device)
            end_time = time.time()
            mean_loss = sum(epoch_loss) / len(epoch_loss)
            print('[Epoch #%d] training loss: %.6f - training time: %.2f seconds' % (epoch + 1, mean_loss, end_time - start_time))
            if self.do_validation and (epoch + 1) % self.valid_steps == 0:
                print(f'--- epoch #{epoch} valid ---')
                logging.info('Start evaluation of validation data')
                self.model.eval()
                with torch.no_grad():
                    if self.model_name in ['ConvE', 'ConvR']:
                        eval_results = eval_conv(valid_loader, self.model, self.device, self.dataset.data)
                        output_eval_conv(eval_results, 'validation')
        if self.do_test:
            print(f'--- test ---')
            logging.info('Start evaluation on test data')
            self.model.eval()
            with torch.no_grad():
                if self.model_name in ['ConvE', 'ConvR']:
                    eval_results = eval_conv(test_loader, self.model, self.device, self.dataset.data)
                    output_eval_conv(eval_results, 'test')
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

    experiment = Experiment(config)
    experiment.train_and_eval()
