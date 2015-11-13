import logging

import numpy as np

format_ = "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=format_)

class Space(object):

    def __init__(self, matrix_, id2row_):

        self.mat = matrix_
        self.id2row = id2row_
        self.create_row2id()

    def create_row2id(self):
        self.row2id = {}
        for idx, word in enumerate(self.id2row):
            if word in self.row2id:
                raise ValueError("Found duplicate word: %s" % (word))
            self.row2id[word] = idx


    @classmethod
    def build(cls, fname, lexicon=None):
        logging.info("Looking up {} words in {}".format(
            len(lexicon) if lexicon else 'all', fname))

        #if lexicon is provided, only data occurring in the lexicon is loaded
        id2row = []
        def filter_lines(f):
            for i,line in enumerate(f):
                # the following three lines contain modifications by Makrai
                if not lexicon and i > 300000:
                    break
                word = line.split(' ')[0]
                if i != 0 and (lexicon is None or word in lexicon):
                    id2row.append(word)
                    yield line

        #get the number of columns
        with open(fname) as f:
            ncols = int(f.readline().strip().split()[1]) + 1 
            m = np.asmatrix(np.loadtxt(filter_lines(f), comments=None,
                                       usecols=range(1,ncols)))
            logging.debug('embedding of shape {} {} read'.format(*m.shape))

        return Space(m, id2row)

    def normalize(self):
        row_norms = np.sqrt(np.multiply(self.mat, self.mat).sum(1))
        row_norms = row_norms.astype(np.double)
        row_norms[row_norms != 0] = np.array(1.0/row_norms[row_norms != 0]).flatten()
        self.mat = np.multiply(self.mat, row_norms)


