# coding: utf-8

import os 
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='RMR', help='name of models') 
    parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of datasets')

    config_dict = {
        'learning_rate': [1e-3], 
        'reg_weight': [0.0001,0.00001],
        'reg_weight': [0.01], 
        'n_layers': [2],
        'reg_weight': [0.01], 
        'gpu_id': 6,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


