import logging
import os

import numpy as np

from space import Space


def prec_at(ranks, cut):
    return len([r for r in ranks if r <= cut])/float(len(ranks))


def get_rank(nn, gold):
    for idx,word in enumerate(nn):
        if word in gold:
            return idx + 1
    return idx + 1


def read_dict(dict_filen, reverse=False, exclude=None, needed=-1):
    logging.info(
    "Reading {} translations from {} ".format(
        needed if needed > 0 else 'all',
        dict_filen))
    if exclude:
        logging.debug('...(other than the {} that were used in training)'.format(
            len(exclude)))
    pairs = dict()
    if not exclude:
        exclude = set()
    with open(dict_filen) as dict_file:
        for i, line in enumerate(dict_file):
            if i == needed:
                logging.debug('read untill line {}'.format(i))
                break
            pair = line.strip().split()
            if reverse:
                pair = tuple(reversed(pair))
            if pair[0] in exclude:
                continue
            pairs[pair[0]] = pair[1]
    logging.debug('{} translations read'.format(len(pairs)))
    return pairs


def apply_tm(sp, tm):

    logging.info("Applying the translation matrix, size of data: %d" %
                 sp.mat.shape[0])
    return Space(sp.mat*tm, sp.id2word)


def get_invocab_trans(sp1, sp2, seed_trans, needed=-1):
    invoc_trans = []
    used_for_train = set()
    collected = 0
    for word1 in sp1.word2id:
        if collected == needed:
            break
        if word1 in seed_trans:
            word2 = seed_trans[word1]
            if word2 in sp2.word2id:
                collected += 1
                invoc_trans.append((word1, word2))
                used_for_train.add(word1)
    logging.info("Using %d word pairs" % collected)
    return invoc_trans, used_for_train


def train_tm(sp1, sp2, seed_trans, train_size):
    seed_trans, used_for_train = get_invocab_trans(sp1, sp2, seed_trans,
                                                needed=train_size)
    els1, els2 = zip(*seed_trans)
    m1 = sp1.mat[[sp1.word2id[el] for el in els1],:]
    m2 = sp2.mat[[sp2.word2id[el] for el in els2],:]
    tm = np.linalg.lstsq(m1, m2, -1)[0]
    return tm, used_for_train


def get_sim_stat(additional, mapped_sr_sp, tg_sp):
    if additional:
        # for each element, computes its rank in the ranked list of
        # similarites. sorting done on the opposite axis (inverse querying)
        rank_mx = np.zeros((tg_sp.mat.shape[0], mapped_sr_sp.mat.shape[0]),
                           dtype='float32')
        logging.debug('rank_mx.shape={}'.format(rank_mx.shape))
        split_size = 10000
        for start in range(0, rank_mx.shape[0], split_size):
            logging.info(
                'Neighbors of {:,}/{:,} source points ranked'.format(start,
                                                                     rank_mx.shape[0]))
            end = min(start + split_size, rank_mx.shape[0])
            #logging.info("Computing cosines...")
            sim_block = - tg_sp.mat[start: end, :]*mapped_sr_sp.mat.T
            sim_block = sim_block.astype('float32')
            #logging.debug("Sorting target space elements...")
            rank_mx[start:end, :] = np.argsort(np.argsort(sim_block, axis=1),
                                               axis=1)
            #logging.info('Combining ranks with cosine similarities...')
            # for each element, the resulting rank is combined with cosine
            # scores.  the effect will be of breaking the ties, because
            # cosines are smaller than 1. sorting done on the standard axis
            # (regular NN querying)
            rank_mx[start:end, :] += sim_block

        logging.info("Sorting by target...")
        rank_mx = np.argsort(rank_mx, axis=0)
        sim_mx = -tg_sp.mat*mapped_sr_sp.mat.T
    else:
        logging.info("Computing cosines and sorting target space elements")
        sim_mx = -tg_sp.mat*mapped_sr_sp.mat.T
        rank_mx = np.argsort(sim_mx, axis=0).A
    return sim_mx, rank_mx


def score(mapped_sr_sp, tg_sp, gold, additional):
    mapped_sr_sp.normalize()
    sim_mx, rank_mx = get_sim_stat(additional, mapped_sr_sp, tg_sp)

    ranks = []
    for i,word1 in enumerate(gold):

        mapped_sr_sp_idx = mapped_sr_sp.word2id[word1]

        # append the top 5 translations
        translations = []
        for j in range(5):
            tg_sp_idx = rank_mx[j, mapped_sr_sp_idx]
            word, score = tg_sp.id2word[tg_sp_idx], -sim_mx[tg_sp_idx, mapped_sr_sp_idx]
            translations.append("\t\t%s:%.3f" % (word, score))

        translations = "\n".join(translations)

        #get the rank of the (highest-ranked) translation
        rnk = get_rank(rank_mx[:,mapped_sr_sp_idx].ravel(),
                        [tg_sp.word2id[el] for el in gold[word1]])
        ranks.append(rnk)

        logging.debug("Id: {}".format(len(ranks)))
        logging.debug("\tSource: {}".format(word1))
        logging.debug("\tTranslation: {}".format(translations))
        logging.debug("\tGold: {}".format(' '.join(gold[word1])))
        logging.debug("\tRank: {}".format(rnk))

    logging.info("Corrected: %s" % str(additional))
    if additional:
        logging.info(
            "{} test and {} additional points, {} in total".format(
                len(gold.keys()), additional, mapped_sr_sp.mat.shape[0]))
    for k in [1,5,10]:
        logging.info("Prec@%d: %.3f" % (k, prec_at(ranks, k)))
    return prec_at(ranks, 1)


def default_output_fn(mx_path, source_fn, target_fn, seed_fn):
    if os.isdir(mx_path):
        mx_path = os.path.join(mx_path, '{}__{}__{}'.format(*[
            os.path.splitext(os.path.basename(fn))
            for fn in [source_fn, target_fn, seed_fn]]))


def get_logger(log_fn):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if log_fn:
        handler = logging.FileHandler(log_fn)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s"))
        logger.addHandler(handler)
