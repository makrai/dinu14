import argparse
import logging

from dinu14.train_tm import train_wrapper
from dinu14.test_tm import test_wrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed')
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--additional', type=int)
    parser.add_argument('--mx')
    parser.add_argument('--test_from_line', type=int)
    parser.add_argument('--train_size', type=int)
    return parser.parse_args()


def train_test_wrapper(args):
    if bool(args['mx']) != bool(args['test_from_line']):
        logging.warning(
            'In order to test an existing translation matrix, both --mx\
            and --test_from_line are needed')
    elif args['mx'] and args['test_from_line']:
        return test_wrapper(args['mx'], args['seed'], args['source'],
                            args['target'], args['additional'],
                            reverse=args['reverse'],
                            first_test=args['test_from_line'])
    mx, last_train = train_wrapper(args['seed'], args['source'],
                                   args['target'], reverse=args['reverse'],
                                   mx_fn=args['mx'], train_size=args['train_size'])
    return test_wrapper(mx, args['seed'], args['source'], args['target'],
                        args['additional'], reverse=args['reverse'],
                        first_test=last_train+1)


if __name__ == '__main__':
    train_test_wrapper(vars(parse_args()))
