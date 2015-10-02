import argparse
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
    parser.add_argument('--mx_fn')
    parser.add_argument('--test_from_line', type=int, default=None)
    parser.add_argument('--train_size', type=int, default=5000)
    return parser.parse_args()


def train_test_wrapper(args):
    if args.mx_fn and os.path.isfile(args.mx_fn):
        if args.test_from_line is None:
            raise Exception(
                'In order to test an existing translation matrix, both' +
                '--mx_fn and --test_from_line are needed')
        elif hasattr(args, 'mx_fn'):
            args.mx_fn = '{}.txt'.format(args.mx_fn)
            return MxTester(args).test_wrapper()
    else:
        mx_fn, last_train = train_wrapper(
            args.seed_fn, args.source_fn, args.target_fn,
            reverse=args.reverse, mx_fn=args.mx_fn,
            train_size=args.train_size)
        if hasattr(args, 'mx_fn'):
            args.mx_fn = '{}.txt'.format(args.mx_fn)
        args.test_from_line = last_train + 1
        return MxTester(args).test_wrapper()


if __name__ == '__main__':
    train_test_wrapper(parse_args())
