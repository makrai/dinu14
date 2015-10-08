import argparse
import simplejson
import logging

from dinu14.train_test_tm import train_test_wrapper

class LinTransWrapper():
    def __init__(self, job_id):
        format_="%(asctime)s: %(module)s (%(lineno)s) %(levelname)s %(message)s"
        logging.basicConfig(format=format_, level=logging.DEBUG)
        self.langs_conf = simplejson.load(open(
            '/mnt/store/makrai/data/language/config.json'))

    def main(self, params):
        pair = ['de', 'hu']
        sr_code = pair[params['reverse'][0]]
        tg_code = pair[1 - params['reverse'][0]]
        embed_i = [params['de_embed_i'][0], params['hu_embed_i'][0]]
        if params['reverse'][0]:
            embed_i = tuple(reversed(embed_i))
        dict_fn = self.langs_conf['biling'][
            '{}-{}'.format(*sorted([sr_code, tg_code]))][
                params['dict_i'][0]]
        source_fn = self.langs_conf['mono'][sr_code]['embed'][embed_i[0]]
        target_fn = self.langs_conf['mono'][tg_code]['embed'][embed_i[1]]
        dinu_args = argparse.Namespace()
        dinu_args_d = vars(dinu_args)
        dinu_args_d.update({
            'seed_fn': dict_fn.format(*pair),
            'source_fn': source_fn,
            'target_fn': target_fn,
            'reverse': params['reverse'][0]})
        for key in ['additional', 'train_size']:
            dinu_args_d[key] = params[key][0]
        return 1 - train_test_wrapper(dinu_args)


def main(job_id, params):
    return LinTransWrapper(job_id).main(params)

if __name__ == "__main__":
    params = {
        "dict_i": [2],
        "reverse": [0],
        "corp_i": [0],
        "hu_embed_i": [33],
        "train_size": [25000],
        "de_embed_i": [0],
        "additional": [50000]
    }
    main(42, params)
