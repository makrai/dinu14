import argparse
import logging
import os

from dinu14.train_tm import train_wrapper
from dinu14.test_tm import MxTester
from dinu14.utils import get_logger, default_output_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_fn')
    parser.add_argument('source_fn')
    parser.add_argument('target_fn')
    parser.add_argument('-r', '--reverse', action='store_true')
    parser.add_argument('--additional', type=int)
    parser.add_argument('--mx_path',
                        help='directory or file name without extension',
                        default='/mnt/store/makrai/project/multiwsi/trans-mx/')
    parser.add_argument('--train_size', type=int, default=5000)
    parser.add_argument('--log-file', dest='log_fn')
    parser.add_argument('--coverage', action='store_true',
                        dest='coverage')
    parser.add_argument('--translate-non-supervised', action='store_true',
                        dest='transt_non_supvisd')
    return parser.parse_args()


def train_test_wrapper(args):
    get_logger(args.log_fn)
    args.mx_path = default_output_fn(args.mx_path, args.seed_fn,
                                     args.source_fn, args.target_fn) 
    if (os.path.isfile('{}.npy'.format(args.mx_path)) and 
            os.path.isfile('{}.train_wds'.format(args.mx_path))):
        logging.info('Testing {}...'.format(args.mx_path))
        return MxTester(args).test_wrapper()
    else:
        mx, used_for_train = train_wrapper(args.seed_fn, args.source_fn,
                                           args.target_fn,
                                           reverse=args.reverse,
                                           mx_path=args.mx_path,
                                           train_size=args.train_size)
        args.mx_path = None
        logging.info('Testing...')
        return MxTester(args, tr_mx=mx,
                        exclude_from_test=used_for_train).test_wrapper()


if __name__ == '__main__':
    train_test_wrapper(parse_args())
