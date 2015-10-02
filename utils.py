import logging

import numpy as np

from space import Space


def prec_at(ranks, cut):
    return len([r for r in ranks if r <= cut])/float(len(ranks))

def get_rank(nn, gold):
    for idx,word in enumerate(nn):
        if word in gold:
            return idx + 1
    return idx + 1


def read_dict(dict_filen, reverse=False, skiprows=0, needed=-1):
    logging.info("Reading: {} from line #{}".format(dict_filen, skiprows))
    pairs = []
    with open(dict_filen) as dict_file:
        for _ in range(skiprows):
            dict_file.readline()
        for i, line in enumerate(dict_file):
            if i == needed:
                break
            pair = line.strip().split()
            if reverse:
                pair = reversed(pair)
            pairs.append(tuple(pair))
    return pairs


def apply_tm(sp, tm):

    logging.info("Applying the translation matrix, size of data: %d" %
                 sp.mat.shape[0])
    return Space(sp.mat*tm, sp.id2row)

def get_valid_data(sp1, sp2, data, needed=-1):
    vdata = []
    collected = 0
    line_i = -1
    for i, (el1,el2) in enumerate(data):
        if el1 in sp1.row2id and el2 in sp2.row2id:
            collected += 1
            vdata.append((el1, el2))
            if collected == needed:
                line_i = i
                break
    logging.info("Using %d word pairs" % collected)
    logging.info("Seed exploited until line #{}".format(line_i))
    return vdata, line_i


def train_tm(sp1, sp2, data, train_size):

    data, last_train = get_valid_data(sp1, sp2, data, needed=train_size)

    els1, els2 = zip(*data)
    m1 = sp1.mat[[sp1.row2id[el] for el in els1],:]
    m2 = sp2.mat[[sp2.row2id[el] for el in els2],:]

    tm = np.linalg.lstsq(m1, m2, -1)[0]

    return tm, last_train


def score(sp1, sp2, gold, additional):

    sp1.normalize()

    logging.info("Computing cosines and sorting target space elements")
    sim_mat = -sp2.mat*sp1.mat.T

    logging.debug('')
    if additional:
        #for each element, computes its rank in the ranked list of
        #similarites. sorting done on the opposite axis (inverse querying)
        srtd_idx = np.argsort(np.argsort(sim_mat, axis=1), axis=1)

        #for each element, the resulting rank is combined with cosine scores.
        #the effect will be of breaking the ties, because cosines are smaller
        #than 1. sorting done on the standard axis (regular NN querying)
        srtd_idx = np.argsort(srtd_idx + sim_mat, axis=0)
    else:
        srtd_idx = np.argsort(sim_mat, axis=0)

    logging.debug('')
    ranks = []
    for i,el1 in enumerate(gold.keys()):

        sp1_idx = sp1.row2id[el1]

        # append the top 5 translations
        translations = []
        for j in range(5):
            sp2_idx = srtd_idx[j, sp1_idx]
            word, score = sp2.id2row[sp2_idx], -sim_mat[sp2_idx, sp1_idx]
            translations.append("\t\t%s:%.3f" % (word, score))

        translations = "\n".join(translations)

        #get the rank of the (highest-ranked) translation
        rnk = get_rank(srtd_idx[:,sp1_idx].A.ravel(),
                        [sp2.row2id[el] for el in gold[el1]])
        ranks.append(rnk)

        logging.debug("\nId: {}".format(len(ranks)))
        logging.debug("\tSource: {}".format(el1))
        logging.debug("\nTranslation: {}".format(translations))
        logging.debug("\tGold: {}".format(gold[el1]))
        logging.debug("\tRank: {}".format(rnk))

    logging.info("Corrected: %s" % str(additional))
    if additional:
        logging.info(
            "Total extra elements, Test({}) + Additional:{}".format(
                len(gold.keys()), sp1.mat.shape[0]))
    for k in [1,5,10]:
        logging.info("Prec@%d: %.3f" % (k, prec_at(ranks, k)))
    return prec_at(ranks, 1)
