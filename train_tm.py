import argparse
import logging
import pickle

import numpy as np

from space import Space
from utils import read_dict, train_tm, default_output_fn

def parse_args():
    parser = argparse.ArgumentParser(
        description="Given train data (pairs of words and their translation),\
        source language and target language vectors, it outputs a translation\
        matrix between source and target spaces.")
    parser.add_argument('--mx_path',
                        help='directory or file name without extension',
                        default='/mnt/store/makrai/project/multiwsi/trans-mx/')
    parser.add_argument(
        'seed_fn',
        help="train dictionary, list of word pairs (space separated words,\
        one word pair per line")
    parser.add_argument(
        'source_fn',
        help="vectors in source language. Space-separated, with string\
        identifier as first column (dim+1 columns, where dim is the\
        dimensionality of the space")
    parser.add_argument(
        'target_fn',
        help="vectors in target language")
    parser.add_argument('--reverse', action='store_true')
    return parser.parse_args()


def train_wrapper(seed_fn, source_fn, target_fn, reverse=False, mx_path=None,
                  train_size=5000):
    logging.info("Training...")
    seed_trans = read_dict(seed_fn, reverse=reverse)

    #we only need to load the vectors for the words in the training data
    #semantic spaces contain additional words
    source_words = set(seed_trans.iterkeys())
    target_words = set().union(*seed_trans.itervalues())

    source_sp = Space.build(source_fn, lexicon=source_words)
    source_sp.normalize()

    target_sp = Space.build(target_fn, lexicon=target_words)
    target_sp.normalize()

    logging.info("Learning the translation matrix")
    tm, used_for_train = train_tm(source_sp, target_sp, seed_trans, train_size)

    mx_path = default_output_fn(mx_path, seed_fn, source_fn, target_fn,)
    logging.info("Saving the translation matrix to {}".format(mx_path))
    np.save('{}.npy'.format(mx_path), tm)
    pickle.dump(used_for_train, open('{}.train_wds'.format(mx_path),
                                     mode='w'))

    return tm, used_for_train


if __name__ == '__main__':
    args = parse_args()
    format_ = "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format_)
    train_wrapper(args.seed_fn, args.source_fn, args.target_fn,
                  reverse=args.reverse, mx_path=args.mx_path)
