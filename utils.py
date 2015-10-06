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
    line_i = -1
    with open(dict_filen) as dict_file:
        for _ in range(skiprows):
            dict_file.readline()
        for i, line in enumerate(dict_file):
            if i == needed:
                line_i = i
                break
            pair = line.strip().split()
            if reverse:
                pair = reversed(pair)
            pairs.append(tuple(pair))
    logging.info("Seed exploited until line #{}".format(line_i))
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


def score(mapped_sr_sp, tg_sp, gold, additional):

    mapped_sr_sp.normalize()

    logging.info("Computing cosines ")
    sim_mx = -tg_sp.mat*mapped_sr_sp.mat.T

    if additional:
        logging.info("Sorting target space elements")
        #for each element, computes its rank in the ranked list of
        #similarites. sorting done on the opposite axis (inverse querying)
        rank_mx = np.zeros(sim_mx.shape)
        split_size = 10000
        for start in range(0, sim_mx.shape[0], split_size):
            end = min(start + split_size, sim_mx.shape[0])
            logging.info(
                'neighbors of {:,}/{:,} source points ranked'.format(
                    start, sim_mx.shape[0]))
            rank_mx[start:end, :] = np.argsort(np.argsort(
                sim_mx[start:end, :], axis=1), axis=1)

        logging.info('Combining ranks with cosine similarities...')
        #for each element, the resulting rank is combined with cosine scores.
        #the effect will be of breaking the ties, because cosines are smaller
        #than 1. sorting done on the standard axis (regular NN querying)
        rank_mx = np.argsort(rank_mx + sim_mx, axis=0)
    else:
        rank_mx = np.argsort(sim_mx, axis=0)

    ranks = []
    for i,el1 in enumerate(gold):

        mapped_sr_sp_idx = mapped_sr_sp.row2id[el1]

        # append the top 5 translations
        translations = []
        for j in range(5):
            tg_sp_idx = rank_mx[j, mapped_sr_sp_idx]
            word, score = tg_sp.id2row[tg_sp_idx], -sim_mx[tg_sp_idx, mapped_sr_sp_idx]
            translations.append("\t\t%s:%.3f" % (word, score))

        translations = "\n".join(translations)

        #get the rank of the (highest-ranked) translation
        rnk = get_rank(rank_mx[:,mapped_sr_sp_idx].A.ravel(),
                        [tg_sp.row2id[el] for el in gold[el1]])
        ranks.append(rnk)

        logging.debug("Id: {}".format(len(ranks)))
        logging.debug("\tSource: {}".format(el1))
        logging.debug("\tTranslation: {}".format(translations))
        logging.debug("\tGold: {}".format(' '.join(gold[el1])))
        logging.debug("\tRank: {}".format(rnk))

    logging.info("Corrected: %s" % str(additional))
    if additional:
        logging.info(
            "{} test and {} additional points, {} in total".format(
                len(gold.keys()), additional, mapped_sr_sp.mat.shape[0]))
    for k in [1,5,10]:
        logging.info("Prec@%d: %.3f" % (k, prec_at(ranks, k)))
    return prec_at(ranks, 1)
