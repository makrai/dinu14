import argparse
import logging
import os

from dinu14.train_tm import train_wrapper
from dinu14.test_tm import MxTester, get_logger

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
    parser.add_argument('--log-file', dest='log_fn') 
    return parser.parse_args()


def train_test_wrapper(args):
    get_logger(args.log_fn)
    if 'mx_fn' not in args:
        args.mx_fn = None
    if args.mx_fn and os.path.isfile(args.mx_fn) and args.test_from_line:
        return MxTester(args).test_wrapper()
    else:
        mx, last_train = train_wrapper(args.seed_fn, args.source_fn,
                                       args.target_fn, reverse=args.reverse,
                                       mx_fn=args.mx_fn,
                                       train_size=args.train_size)
        args.test_from_line = last_train + 1
        args.mx_fn = None
        return MxTester(args, tr_mx=mx).test_wrapper()


if __name__ == '__main__':
    format_ = "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format_)
    train_test_wrapper(parse_args())
