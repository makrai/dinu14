import argparse
import collections
import random
import logging

import numpy as np

from space import Space
from dinu14.utils import read_dict, apply_tm, score, get_valid_data

class MxTester():
    def __init__(self, args):
        self.tm = args['mx_fn']
        self.additional = args['additional'] 
        self.main(args)

    def main(self, args):
        self.get_logger(args['log_fn'])

        if isinstance(self.tm, basestring):
            logging.info("Loading the translation matrix")
            self.tm = np.loadtxt(self.tm)

        logging.info("Reading the test data")
        test_data = read_dict(args['seed_fn'], reverse=args['reverse'],
                              skiprows=args['first_test'], needed=1000)

        source_sp = self.build_source_sp(args['source_fn'], test_data)


        target_sp = Space.build(args['target_fn'])
        target_sp.normalize()

        logging.info(
            "Translating all the elements loaded in the source space")
        mapped_source_sp = apply_tm(source_sp, self.tm)

        logging.info("Retrieving translations")
        test_data, _ = get_valid_data(source_sp, target_sp, test_data)

        #turn test data into a dictionary (a word can have mutiple translation)
        gold = collections.defaultdict(set)
        gold.update(dict(test_data))

        if args['mapped_vecs']:
            logging.info("Printing mapped vectors: %s" % args['mapped_vecs'])
            np.savetxt("%s.vecs.txt" % args['mapped_vecs'], mapped_source_sp.mat)
            np.savetxt(
                "%s.wds.txt" % args['mapped_vecs'], mapped_source_sp.id2row, fmt="%s")

        return score(mapped_source_sp, target_sp, gold, self.additional)

    def get_logger(self, log_fn):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        if log_fn:
            handler = logging.FileHandler(log_fn, encoding='utf8')
        else:
            handler = logging.StreamHandler() 
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"))
        logger.addHandler(handler)

    def build_source_sp(self, source_file, test_data):
        #in the _source_ space, we only need to load vectors for the words in test.
        #semantic spaces may contain additional words, ALL words in the _target_
        #space are used as the search space
        source_words, _ = zip(*test_data)
        source_words = set(source_words)
        if self.additional:
            #read all the words in the space
            lexicon = set(np.loadtxt(source_file, skiprows=1, dtype=str,
                                        comments=None, usecols=(0,)).flatten())
            #the max number of additional+test elements is bounded by the size
            #of the lexicon
            self.additional = min(self.additional, len(lexicon) - len(source_words))
            #we sample additional elements that are not already in source_words
            random.seed(100)
            logging.info(self.additional)
            lexicon = random.sample(list(lexicon.difference(source_words)),
                                    self.additional)

            #load the source space
            source_sp = Space.build(source_file, source_words.union(set(lexicon)))
        else:
            source_sp = Space.build(source_file, source_words) 
        source_sp.normalize()
        return source_sp


def parse_args():
    parser = argparse.ArgumentParser(
    description="Given a translation matrix, test data (words and their\
        translations) and source and target language vectors, it returns\
        translations of source test words and computes Top N accuracy.",
        epilog='\n\
        Example:\n\
        1) Retrieve translations with standard nearest neighbour retrieval\n\
        \n\
        python test_tm.py tm.txt test_data.txt ENspace.txt ITspace.txt\n\
        \n\
        2) "Corrected" retrieval (GC). Use additional 2000 source space\n\
        elements to correct for hubs (words that appear as the nearest\n\
        neighbours of many points))\n\
        \n\
        python -c 2000 test_tm.py tm.txt test_data.txt ENspace.txt ITspace.txt')
    parser.add_argument('mx_fn', help='translation matrix')
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
    parser.add_argument(
        '--additional', type=int,
        help='Number of elements (additional to test data) to be used with\
        Global Correction (GC) strategy.')
    parser.add_argument('-o', '--log-file', dest='log_fn') 
    parser.add_argument(
        '--mapped_vecs', 
        help='File prefix. It prints the vectors obtained after the\
        translation matrix is applied (.vecs.txt and .wds.txt).')
    parser.add_argument('--test-from-line', dest='first_test', type=int, 
                        default=0, help='intedex from 0')
    return parser.parse_args()


if __name__ == "__main__":
    MxTester(vars(parse_args())).main()
