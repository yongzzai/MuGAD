import warnings
warnings.filterwarnings('ignore')


import os
import time
import datetime
import logging

from dataset.GraphDataset import GraphDataset
from model.MuGAD import MuGAD
from utils.eval import evaluation
from utils.utils import generate_label_indice, parse_, get_model_args
from utils.config import EVENTLOG_DIR


def run(dataset, label_indices, model):
    model.fit(dataset, label_indices)
    trace_level_labels, trace_level_preds, event_level_labels, event_level_preds = model.detect(dataset, label_indices)
    f1_trace, f1_event = evaluation(trace_level_labels, trace_level_preds, event_level_labels, event_level_preds)
    logging.info(f"Trace level F1-score: {f1_trace}")
    logging.info(f"Event level F1-score: {f1_event}")
    return f1_trace, f1_event


import argparse
parser = argparse.ArgumentParser(description='MuGAD')
parser.add_argument('--hidden_dim', type=int)
parser.add_argument('--num_conv_layer', type=int)
parser.add_argument('--pretrain_epoch', type=int)
parser.add_argument('--adapt_epoch', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--thres', type=float)
parser.add_argument('--beta', type=float)
parser.add_argument('--k', type=int)
parser.add_argument('--lp', type=float)
args = parser.parse_args()
model_args = get_model_args(args)

if __name__ == '__main__':

    pid = os.getpid()
    print(f'Process ID: {pid}')
    print("====================================================")
    log_filename = f'results/{datetime.datetime.now().strftime("%m-%d_%H-%M-%S")}-lr({args.learning_rate})-batch({args.batch_size})[{pid}].log'
        
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s')
    logging.info("====================================================")
    logging.info(f"Exp with params:")
    logging.info("")
    logging.info(f"hidden_dim: {args.hidden_dim}, num_conv_layer: {args.num_conv_layer}")
    logging.info(f"pretrain_epoch: {args.pretrain_epoch}, adapt_epoch: {args.adapt_epoch}")
    logging.info(f"learning_rate: {args.learning_rate}, batch_size: {args.batch_size}")
    logging.info(f"thres: {args.thres}, beta: {args.beta}, k: {args.k}, label_perc: {args.lp}")
    logging.info("====================================================")
    

    dataset_names = [name.split('.')[0] for name in os.listdir(EVENTLOG_DIR) if name.endswith('.csv')]

    for name in dataset_names:
        
        logging.info(f"Running experiment on dataset: {name}")

        dataset = GraphDataset(dataset_name=name)\
                        
        pi_p = parse_(name)
        model_args['pi_p'] = pi_p
        label_indices = generate_label_indice(dataset=dataset, label_perc=args.lp)

        start_time = time.time()
        model = MuGAD(**model_args)
        f1_trace, f1_event = run(dataset, label_indices, model)
        end_time = time.time()

        print(f"Trace level F1-score: {f1_trace}")
        print(f"Event level F1-score: {f1_event}")

        logging.info(f"Time taken: {end_time - start_time} seconds")

    logging.info("Experiment completed.")
    print(f'Process ID: {pid}')
