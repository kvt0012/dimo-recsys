import os
import argparse
import multiprocessing

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

import pickle

import mrecsys.factorization
from mrecsys.utils.model_selection import EvalResults
from mrecsys.utils.dataset import load_latest_interactions

os.chdir(os.path.dirname(__file__))

NUM_SAMPLES = 1
N_ITER = list(range(15, 15 + 1, 5))
FACTORS = list(range(16, 16 + 1, 8))
nproc = multiprocessing.cpu_count()


def _train_als(hyperparameters, train):
    h = hyperparameters
    model = AlternatingLeastSquares(factors=h['factors'],
                                    iterations=h['n_iter'],
                                    num_threads=nproc)

    model.fit(train)
#    test_eval = {'p@k': precision_at_k(model, train.T.tocsr(), factorization.T.tocsr(), K=10)}
#    val_eval = {'p@k': precision_at_k(model, train.T.tocsr(), validation.T.tocsr(), K=10)}
    return model


def _train_bpr(hyperparameters, train):
    h = hyperparameters
    model = BayesianPersonalizedRanking(factors=h['factors'],
                                        iterations=h['n_iter'],
                                        num_threads=nproc)

    model.fit(train)
#    test_eval = {'p@k': precision_at_k(model, train.T.tocsr(), factorization.T.tocsr(), K=10)}
#    val_eval = {'p@k': precision_at_k(model, train.T.tocsr(), validation.T.tocsr(), K=10)}
    return model


def _train_lmf(hyperparameters, train):
    h = hyperparameters
    model = LogisticMatrixFactorization(factors=h['factors'],
                                        iterations=h['n_iter'],
                                        num_threads=nproc)

    model.fit(train)
#    test_eval = {'p@k': precision_at_k(model, train.T.tocsr(), factorization.T.tocsr(), K=10)}
#    val_eval = {'p@k': precision_at_k(model, train.T.tocsr(), validation.T.tocsr(), K=10)}
    return model


result_path = mrecsys.factorization.__result_path__
model_path = mrecsys.factorization.__model_path__


def run(interactions=None, time_code=None, model_type='bpr'):

    if interactions is None or time_code is None:
        interactions, time_code, _, _ = load_latest_interactions()

    interactions = interactions.tocsr().T.tocsr()

    if model_type == 'als':
        train_fnc = _train_als
    elif model_type == 'bpr':
        train_fnc = _train_bpr
    elif model_type == 'lmf':
        train_fnc = _train_lmf
    else:
        raise ValueError('Unknown model type')

    tuned_results = EvalResults(os.path.join(result_path, 'tuning/{}_results_{}.txt'.format(model_type, time_code)))
    params = tuned_results.best("p@k")
    print('Training {} model with params {}'.format(model_type, params))
    model = train_fnc(params, interactions)
    pickle.dump(model, open(os.path.join(model_path, '{}_model_{}.pkl'.format(model_type, time_code)), 'wb'))

    # train_pk = precision_at_k(model, interactions, K=10, num_threads=nproc)
    # trained_result = EvalResults(os.path.join(mrecsys.factorization.__result_path__,
    #                                           'trained/{}_result.txt'.format(model_type)))
    # trained_result.save(params, test_eval=train_pk, time_code=time_code)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='define the network (als / bpr / lmf)')
    args = parser.parse_args()
    model_type = args.model
    if model_type is None:
        model_type = input('Enter model type (als / bpr / lmf): ')
    interactions, time_code, _, _ = load_latest_interactions()

    run(interactions, time_code, model_type)
