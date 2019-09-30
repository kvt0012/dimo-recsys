import os
import shutil
import argparse

import torch
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet
from spotlight.evaluation import sequence_mrr_score

import mrecsys.sequence
from mrecsys.utils.model_selection import EvalResults
from mrecsys.utils.dataset import load_latest_interactions

CUDA = (os.environ.get('CUDA') is not None or shutil.which('nvidia-smi') is not None)
result_path = mrecsys.sequence.__result_path__
model_path = mrecsys.sequence.__model_path__

os.chdir(os.path.dirname(__file__))


def _train_lstm(hyperparameters, train, random_state):
    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)
    model.fit(train, verbose=True)
    return model


def _train_cnn(hyperparameters, train, random_state):
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
    return model


def _train_pooling(hyperparameters, train, random_state):
    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)
    model.fit(train, verbose=True)
    return model


def run(interactions=None, time_code=None, model_type='lstm'):
    print('CUDA', CUDA)
    random_state = mrecsys.sequence.__random_state__
    if interactions is None or time_code is None:
        interactions, time_code, _, _ = load_latest_interactions()

    # train, rest = user_based_train_test_split(interactions, random_state=random_state, test_size=test_size)
    # factorization, validation = user_based_train_test_split(rest, random_state=random_state)
    # train_sq = train.to_sequence()
    # test_sq = factorization.to_sequence()
    # validation_sq = validation.to_sequence()
    print('Training {} with {}'.format(model_type, interactions))
    train_sq = interactions.to_sequence()

    if model_type == 'lstm':
        train_fnc = _train_lstm
    elif model_type == 'cnn':
        train_fnc = _train_cnn
    elif model_type == 'pooling':
        train_fnc = _train_pooling
    else:
        raise ValueError('Unknown model type')

    tuned_results = EvalResults(os.path.join(result_path, 'tuning/{}_results.txt'.format(model_type, time_code)))
    params = tuned_results.best('mrr')
    print('Training {} model with: {} '.format(model_type, params))

    model = train_fnc(params, train_sq, random_state)
    torch.save(model, os.path.join(model_path, '{}_model_{}.pt'.format(model_type, time_code)))

    # test_mrr, val_mrr = sequence_mrr_score(model, test_sq), sequence_mrr_score(model, validation_sq)
    train_mrr = sequence_mrr_score(model, train_sq)
    trained_result = EvalResults(os.path.join(mrecsys.sequence.__result_path__,
                                              'trained/{}_result.txt'.format(model_type)))
    trained_result.save(params, test_eval=train_mrr, time_code=time_code)
    # trained_result.save(params, test_mrr, val_mrr, curr_datetime)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='define the network (cnn / lstm / pooling)')
    args = parser.parse_args()
    model_type = args.model

    if model_type is None:
        model_type = input('Enter model type (cnn / lstm / pooling): ')
    interactions, time_code, _, _ = load_latest_interactions()
    run(interactions=interactions, time_code=time_code, model_type=model_type)


if __name__ == '__main__':
    main()
