import os
import argparse
import multiprocessing

from sklearn.model_selection import ParameterSampler
from implicit.evaluation import precision_at_k, train_test_split
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

import mrecsys.factorization
from mrecsys.utils.model_selection import EvalResults
from mrecsys.utils.dataset import load_latest_interactions

os.chdir(os.path.dirname(__file__))

NUM_SAMPLES = 100
N_ITER = list(range(15, 45 + 1, 5))
FACTORS = list(range(8, 64 + 1, 8))

nproc = multiprocessing.cpu_count()


def sample_factorization_hyperparameters(num):
    space = {
        'n_iter': N_ITER,
        'factors': FACTORS,
    }
    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=mrecsys.factorization.__random_state__)
    for params in sampler:
        yield params


def evaluate_als_model(hyperparameters, train, test, validation):
    h = hyperparameters

    model = AlternatingLeastSquares(factors=h['factors'],
                                    iterations=h['n_iter'],
                                    num_threads=nproc)

    model.fit(train)
    test_eval = {'p@k': precision_at_k(model, train.T.tocsr(), test.T.tocsr(), K=10)}
    val_eval = {'p@k': precision_at_k(model, train.T.tocsr(), validation.T.tocsr(), K=10)}
    return test_eval, val_eval


def evaluate_bpr_model(hyperparameters, train, test, validation):
    h = hyperparameters

    model = BayesianPersonalizedRanking(factors=h['factors'],
                                        iterations=h['n_iter'],
                                        num_threads=nproc)

    model.fit(train)
    test_eval = {'p@k': precision_at_k(model, train.T.tocsr(), test.T.tocsr(), K=10)}
    val_eval = {'p@k': precision_at_k(model, train.T.tocsr(), validation.T.tocsr(), K=10)}
    return test_eval, val_eval


def evaluate_lmf_model(hyperparameters, train, test, validation):
    h = hyperparameters

    model = LogisticMatrixFactorization(factors=h['factors'],
                                        iterations=h['n_iter'],
                                        num_threads=nproc)

    model.fit(train)
    test_eval = {'p@k': precision_at_k(model, train.T.tocsr(), test.T.tocsr(), K=10)}
    val_eval = {'p@k': precision_at_k(model, train.T.tocsr(), validation.T.tocsr(), K=10)}
    return test_eval, val_eval


def tuning(train, test, validation, model_type, time_code):

    sample_fnc = sample_factorization_hyperparameters
    if model_type == 'als':
        eval_fnc = evaluate_als_model
    elif model_type == 'bpr':
        eval_fnc = evaluate_bpr_model
    elif model_type == 'lmf':
        eval_fnc = evaluate_lmf_model
    else:
        raise ValueError('Unknown model type')

    results = EvalResults(os.path.join(mrecsys.factorization.__result_path__, 'tuning/{}_results.txt'.format(model_type, time_code)))
    best_result = results.best('p@k')
    print('Best {} result by p@k: {}'.format(model_type, best_result))

    for hyperparameters in sample_fnc(NUM_SAMPLES):
        if hyperparameters in results:
            continue

        try:
            print('Evaluating {}'.format(hyperparameters))

            (test_eval, val_eval) = eval_fnc(hyperparameters,
                                             train,
                                             test,
                                             validation)
            print('test_eval:', test_eval)
            print('val_eval:', val_eval)
            results.save(hyperparameters, test_eval, val_eval)
        except KeyboardInterrupt as e:
            raise e
        except:
            pass

    return results


def run(model_type=None):
    if model_type is None:
        model_type = input('Enter model type (als / bpr / lmf): ')
    interactions, time_code, _, _ = load_latest_interactions()
    interactions = interactions.tocsr().T.tocsr()
    train, rest = train_test_split(interactions)
    test, validation = train_test_split(rest)
    # print('Split into \n {} and \n {} and \n {}.'.format(train, factorization, validation))

    tuning(train, test, validation, model_type, time_code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='define the network (als / bpr / lmf)')
    args = parser.parse_args()
    model_type = args.model
    run(model_type)
