import argparse
import logging

from dinu14.train_tm import train_wrapper
from dinu14.test_tm import MxTester
from dinu14.utils import get_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_fn')
    parser.add_argument('source_fn')
    parser.add_argument('target_fn')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--additional', type=int)
    parser.add_argument('--mx_fn', help='including extension', default=None)
    parser.add_argument('--train_size', type=int, default=5000)
    parser.add_argument('--log-file', dest='log_fn')
    parser.add_argument('--coverage', action='store_true',
                        dest='coverage')
    parser.add_argument('--translate-non-supervised', action='store_true',
                        dest='transt_non_supvisd')

    return parser.parse_args()


def train_test_wrapper(args):
    get_logger(args.log_fn)
    mx, used_for_train = train_wrapper(args.seed_fn, args.source_fn,
                                       args.target_fn, reverse=args.reverse,
                                       mx_fn=args.mx_fn,
                                       train_size=args.train_size)
    args.mx_fn = None
    logging.info('testing...')
    return MxTester(args, tr_mx=mx,
                    exclude_from_test=used_for_train).test_wrapper()


if __name__ == '__main__':
    train_test_wrapper(parse_args())
