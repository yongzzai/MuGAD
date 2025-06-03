import numpy as np
import random

def generate_label_indice(dataset, label_perc):

    anomaly_indices = [idx for idx, g in enumerate(dataset.graphs) if g.trace_y == 1]

    num_labelled = int(np.ceil(len(anomaly_indices)*label_perc))

    labeled_indices = random.sample(anomaly_indices, k=num_labelled)

    for idx in labeled_indices:
        assert dataset.graphs[idx].trace_y == 1, f"Expected 1, but got {dataset.graphs[idx].trace_y}"

    return labeled_indices


def parse_(name):
    match name.split('-(')[1].split(')')[0]:
        case '5':
            return 0.05
        case '10':
            return 0.1
        case '15':
            return 0.15
        case '20':
            return 0.2
        case '25':
            return 0.25
        case '30':
            return 0.3


def get_model_args(args):
    return {'hidden_dim': args.hidden_dim, 'num_conv_layer': args.num_conv_layer,
            'pretrain_epoch': args.pretrain_epoch, 'adapt_epoch': args.adapt_epoch,
            'learning_rate': args.learning_rate, 'batch_size': args.batch_size,
            'thres': args.thres, 'beta': args.beta, 'k': args.k}
