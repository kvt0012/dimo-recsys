import shutil
import os
import argparse

from spotlight.cross_validation import user_based_train_test_split
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet
from spotlight.evaluation import sequence_mrr_score

from sklearn.model_selection import ParameterSampler

import mrecsys.sequence
from mrecsys.utils.model_selection import EvalResults
from mrecsys.utils.dataset import load_latest_interactions

os.chdir(os.path.dirname(__file__))

CUDA = (os.environ.get('CUDA') is not None or shutil.which('nvidia-smi') is not None)

NUM_SAMPLES = 100
LOSSES = ['bpr', 'pointwise', 'hinge', 'adaptive_hinge']
N_ITER = list(range(35, 75, 5))
METRICS = ['mrr', 'p@k', 'r@k', 'rmse']


def sample_cnn_hyperparameters(random_state, num):
    space = {
        'n_iter': N_ITER,
        'loss': LOSSES,
        'kernel_width': [3, 5, 7],
        'num_layers': list(range(1, 10)),
        'dilation_multiplier': [1, 2],
        'nonlinearity': ['tanh', 'relu'],
        'residual': [True, False]
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:
        params['dilation'] = list(params['dilation_multiplier'] ** (i % 8)
                                  for i in range(params['num_layers']))
        yield params


def sample_lstm_hyperparameters(random_state, num):
    space = {
        'n_iter': N_ITER,
        'loss': LOSSES,
    }
    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)
    for params in sampler:
        yield params


def sample_immf_hyperparameters(random_state, num):
    space = {
        'loss': LOSSES,
        'n_iter': N_ITER,
    }
    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)
    for params in sampler:
        yield params


def evaluate_cnn_model(hyperparameters, train, test, validation, random_state):
    h = hyperparameters

    net = CNNNet(train.num_items,
                 kernel_width=h['kernel_width'],
                 dilation=h['dilation'],
                 num_layers=h['num_layers'],
                 nonlinearity=h['nonlinearity'],
                 residual_connections=h['residual'])

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation=net,
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_eval = {}
    test_eval['mrr'] = sequence_mrr_score(model, test).mean()

    val_eval = {}
    val_eval['mrr'] = sequence_mrr_score(model, validation).mean()

    return test_eval, val_eval


def evaluate_lstm_model(hyperparameters, train, test, validation, random_state):
    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='lstm',
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_eval = {}
    test_eval['mrr'] = sequence_mrr_score(model, test).mean()

    val_eval = {}
    val_eval['mrr'] = sequence_mrr_score(model, validation).mean()

    return test_eval, val_eval


def evaluate_pooling_model(hyperparameters, train, test, validation, random_state):
    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='pooling',
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_eval = {}
    test_eval['mrr'] = sequence_mrr_score(model, test).mean()

    val_eval = {}
    val_eval['mrr'] = sequence_mrr_score(model, validation).mean()

    return test_eval, val_eval


def tuning(train, test, validation, random_state, model_type, time_code):

    if model_type != 'immf':
        train = train.to_sequence()
        test = test.to_sequence()
        validation = validation.to_sequence()

    if model_type == 'cnn':
        eval_fnc, sample_fnc = (evaluate_cnn_model,
                                sample_cnn_hyperparameters)
    elif model_type == 'lstm':
        eval_fnc, sample_fnc = (evaluate_lstm_model,
                                sample_lstm_hyperparameters)
    elif model_type == 'pooling':
        eval_fnc, sample_fnc = (evaluate_pooling_model,
                                sample_lstm_hyperparameters)
    else:
        raise ValueError('Unknown model type')

    results = EvalResults(os.path.join(mrecsys.sequence.__result_path__, 'tuning/{}_result_{}.txt'.format(model_type, time_code)))
    best_results = {}
    for metric in METRICS:
        if results.best(metric) is not None:
            best_results[metric] = results.best(metric)
            print('Best {} result by {}: {}'.format(model_type, metric, best_results[metric]))

    for hyperparameters in sample_fnc(random_state, NUM_SAMPLES):
        if hyperparameters in results:
            continue

        try:
            print('Evaluating {}'.format(hyperparameters))

            (test_eval, val_eval) = eval_fnc(hyperparameters,
                                             train,
                                             test,
                                             validation,
                                             random_state)
            print('test_eval:', test_eval)
            print('val_eval:', val_eval)
            results.save(hyperparameters, test_eval, val_eval)
        except KeyboardInterrupt as e:
            raise e
        except:
            pass

    return results


def run(model_type=None):
    random_state = mrecsys.sequence.__random_state__

    if model_type is None:
        model_type = input('Enter model type (cnn / lstm / pooling): ')
    print('CUDA:', CUDA)
    interactions, time_code, _, _ = load_latest_interactions()
    train, rest = user_based_train_test_split(interactions, random_state=random_state)
    test, validation = user_based_train_test_split(rest, random_state=random_state)
    print('Split into \n {} and \n {} and \n {}.'.format(train, test, validation))

    tuning(train, test, validation, random_state, model_type, time_code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='define the network (cnn / lstm / pooling)')
    args = parser.parse_args()
    model_type = args.model
    run(model_type)
