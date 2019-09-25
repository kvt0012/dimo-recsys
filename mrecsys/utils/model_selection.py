import hashlib
import json
from os import listdir
from os.path import isfile, join
import torch
import pickle


def select_latest_model(dirpath, model_type):
    files = [f for f in listdir(dirpath)
             if (isfile(join(dirpath, f)) and f.find(model_type) > -1 and f.find('model') > -1)]
    if len(files) > 0:
        try:
            time_code = files[-1].split('.')[0].split('_')[2]
            if files[-1].find('.pt') > -1:
                model = torch.load(join(dirpath, files[-1]))
            elif files[-1].find('.pkl') > -1:
                model = pickle.load(open(join(dirpath, files[-1]), 'rb'))
        except FileNotFoundError as e:
            raise e('Model file format is not valid')
        return model, time_code
    raise FileNotFoundError('Not found any models')


def select_model_by_time(time_code, dirpath, model_type, extension='.pt'):
    return torch.load(join(dirpath, '{}_model_{}{}'.format(model_type, time_code, extension)))


class EvalResults:

    def __init__(self, filename):
        self._filename = filename
        open(self._filename, 'a+')

    @staticmethod
    def _hash(x):
        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save(self, hyperparams,
             test_eval, val_eval=None, time_code=None):

        result = {'test': test_eval,
                  'hash': self._hash(hyperparams)}
        if time_code is not None:
            result['time_code'] = time_code
        if val_eval is not None:
            result['validation'] = val_eval

        result.update(hyperparams)
        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best(self, metric, greater_is_better=True):
        try:
            y = -1
            if greater_is_better:
                y = 1
            results = sorted([x for x in self],
                             key=lambda x: (-y) * x['test'][metric])
        except:
            return None

        if results:
            return results[0]
        else:
            return None

    def __getitem__(self, hyperparams):
        params_hash = self._hash(hyperparams)
        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)
                if datum['hash'] == params_hash:
                    del datum['hash']
                    return datum
        raise KeyError

    def __contains__(self, x):
        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):
        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)
                del datum['hash']
                yield datum
