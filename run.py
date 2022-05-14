import argparse
import os
import re

import numpy as np
import torch

from cmds import Train
from parser.utils.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the models under few samples.')
    # subparsers
    subparsers = parser.add_subparsers(title='Commands', dest='mode')

    # subcommands
    subcommands = {'train': Train()}

    for mode, subcommand in subcommands.items():
        subparser = subcommand.create_subparser(subparsers, mode)
        subparser.add_argument('--config',
                               default='configs/crf.ini',
                               help='path to config file')
        subparser.add_argument('--save',
                               default='save/master',
                               help='path to saved files')
        subparser.add_argument('--device',
                               default='0',
                               help='ID of GPU to use')
        subparser.add_argument('--seed',
                               default=0,
                               type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--batch-size',
                               default=5000,
                               type=int,
                               help='batch size')
        subparser.add_argument('--n-buckets',
                               default=32,
                               type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--threads',
                               default=8,
                               type=int,
                               help='max num of threads')
        subparser.add_argument('--embed',
                               default=None,
                               help='embed file')
        subparser.add_argument('--model',
                               default="crf",
                               choices=["crf", "crf_ae", "gaussian_hmm"],
                               type=str,
                               help='type of model')

    # parse args
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    torch.set_num_threads(args.threads)

    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Set the device with ID {args.device} visible")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args = Config(args.config).update(vars(args))
    args.update({"save": re.sub(r"[ \r]", "", args.save.strip())})
    args.fields = os.path.join(args.save, 'fields')
    args.model_path = os.path.join(args.save, f'model')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Run the command in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(args)
