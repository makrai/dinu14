import argparse
import collections
import random
import logging
import os

import numpy as np

from space import Space
from dinu14.utils import read_dict, apply_tm, score, get_valid_data

class MxTester():
    def __init__(self, args, tr_mx=None):
        self.additional = args.additional 
        self.args = args
        self.tr_mx = tr_mx

    def test_wrapper(self):
        if hasattr(self.args, 'log_fn') and self.args.log_fn:
            self.get_logger(self.args.log_fn)

        if self.args.mx_fn:
            if self.tr_mx:
                raise Exception("Translation mx specified amibiguously.")
            else:
                logging.info("Loading the translation matrix")
                _, ext = os.path.splitext(self.args.mx_fn)
                if ext == '.npy':
                    self.tr_mx = np.load(self.args.mx_fn)
                elif ext == '.txt':
                    self.tr_mx = np.loadtxt(self.args.mx_fn)
                else:
                    raise Exception(
                        'Unknown extension for translation matrix: {}'.format(
                            ext))

        test_data = read_dict(self.args.seed_fn, reverse=self.args.reverse,
                              skiprows=self.args.test_from_line, needed=1000)

        source_sp = self.build_source_sp(self.args.source_fn, test_data) 

        target_sp = Space.build(self.args.target_fn)
        target_sp.normalize()

        test_data, _ = get_valid_data(source_sp, target_sp, test_data)

        #turn test data into a dictionary (a word can have mutiple translation)
        gold = collections.defaultdict(set)
        for sr, tg in test_data:
            gold[sr].add(tg)

        logging.info(
            "Mapping all the elements loaded in the source space")
        mapped_source_sp = apply_tm(source_sp, self.tr_mx)
        if hasattr(self.args, 'mapped_vecs') and self.args.mapped_vecs:
            logging.info("Printing mapped vectors: %s" % self.args.mapped_vecs)
            np.savetxt("%s.vecs.txt" % self.args.mapped_vecs, mapped_source_sp.mat)
            np.savetxt(
                "%s.wds.txt" % self.args.mapped_vecs, mapped_source_sp.id2row, fmt="%s")

        return score(mapped_source_sp, target_sp, gold, self.additional)

    def get_logger(self, log_fn):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        if log_fn:
            handler = logging.FileHandler(log_fn)
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"))
        logger.addHandler(handler)

    def build_source_sp(self, source_file, test_data):
        """
        In the _source_ space, we only need to load vectors for the words in test.
        Semantic spaces may contain additional words. 
        All words in the _target_ space are used as the search space
        """
        source_words, _ = zip(*test_data)
        source_words = set(source_words)
        if self.additional:
            #read all the words in the space
            lexicon = set(np.loadtxt(source_file, skiprows=1, dtype=str,
                                        comments=None, usecols=(0,)).flatten())
            #the max number of additional+test elements is bounded by the size
            #of the lexicon
            self.additional = min(self.additional, len(lexicon) - len(source_words))
            random.seed(100)
            logging.info("Sampling {} additional elements".format(self.additional))
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
        python test_tm.py tm.npy test_data.txt ENspace.txt ITspace.txt\n\
        \n\
        2) "Corrected" retrieval (GC). Use additional 2000 source space\n\
        elements to correct for hubs (words that appear as the nearest\n\
        neighbours of many points))\n\
        \n\
        python -c 2000 test_tm.py tm.npy test_data.txt ENspace.txt ITspace.txt')
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
    parser.add_argument('--test-from-line', dest='test_from_line', type=int, 
                        default=0, help='intedex from 0')
    return parser.parse_args()


if __name__ == "__main__":
    MxTester(parse_args()).test_wrapper()
