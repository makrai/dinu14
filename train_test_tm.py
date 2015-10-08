import argparse
import logging
import os

from dinu14.train_tm import train_wrapper
from dinu14.test_tm import MxTester

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_fn')
    parser.add_argument('source_fn')
    parser.add_argument('target_fn')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--additional', type=int)
    parser.add_argument('--mx_fn', help='including extension')
    parser.add_argument('--test_from_line', type=int, default=None)
    parser.add_argument('--train_size', type=int, default=5000)
    return parser.parse_args()


def train_test_wrapper(args):
    if 'mx_fn' not in args:
        args.mx_fn = None
    if args.mx_fn and os.path.isfile(args.mx_fn):
        if args.test_from_line is None:
            raise Exception(
                'In order to test an existing translation matrix, both' +
                '--mx_fn and --test_from_line are needed')
        else: 
            return MxTester(args).test_wrapper()
    else:
        mx, last_train = train_wrapper(
            args.seed_fn, args.source_fn, args.target_fn,
            reverse=args.reverse, mx_fn=args.mx_fn,
            train_size=args.train_size)
        args.test_from_line = last_train + 1
        return MxTester(args, tr_mx=mx).test_wrapper()


if __name__ == '__main__':
    format_ = "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format_)
    train_test_wrapper(parse_args())
