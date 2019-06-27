import numpy as np

import sys
sys.path.append('./')
from train_base import read_info, write_csv
from train_base import read_data as read_raw_data
from model.ngram import NGramModel
from util import argparser
from gp import bayesian_optimisation

results = [['lang', 'order', 'params', 'val_loss']]


def get_data(lang):
    train_loader, _ = get_ngram_data(lang, 'train')
    val_loader, _ = get_ngram_data(lang, 'val')
    test_loader, _ = get_ngram_data(lang, 'test')

    return train_loader, val_loader, test_loader


def get_ngram_data(lang, mode):
    data, idx = read_data(lang, mode)
    return remove_pads(data, mode), idx


def read_data(lang, mode):
    data = read_raw_data(lang, mode)
    data = data[:, :-1]
    idx = data[:, -1]
    return data, idx


def remove_pads(data, mode):
    parsed_data = []
    for datum in data:
        reduced_datum = datum[datum != 0]
        parsed_data += [reduced_datum]

    return parsed_data


def get_model_entropy(model, train_loader, eval_loader, vocab_size, params=None, order=3):
    if model == 'ngram':
        pass
    else:
        raise ValueError("Model not implemented: %s" % model)

    params = params / np.sum(params, keepdims=True)
    model = NGramModel(train_loader, vocab_size, order=order, params=params)
    val_loss = model.get_entropy(eval_loader)

    return val_loss


def sample_loss_getter(lang, train_loader, val_loader, test_loader, token_map, args):
    global count
    vocab_size = len(token_map)
    count = 0

    def sample_loss(hyper_params):
        global results, count
        print(count, hyper_params)
        count += 1

        order = len(hyper_params)
        params = hyper_params

        val_loss = get_model_entropy(args.model, train_loader, val_loader, vocab_size, params=params, order=order)

        results += [[lang, order, params, val_loss]]
        return val_loss

    return sample_loss


def optimize_model_rnd(model, train_loader, val_loader, test_loader, vocab_size, order=3, tests=1):
    if model == 'ngram':
        pass
    else:
        raise ValueError("Model not implemented: %s" % model)

    model = NGramModel(train_loader, vocab_size, order=order)
    entropies = [(model.params, model.get_entropy(val_loader))]

    test_params = np.random.uniform(size=(tests, order))
    test_params = test_params / np.sum(test_params, axis=1, keepdims=True)

    for params in test_params:
        model = NGramModel(train_loader, vocab_size, order=order, params=params)
        entropies += [(model.params, model.get_entropy(val_loader))]

    min_params, val_loss = min(entropies, key=lambda x: x[1])

    model = NGramModel(train_loader, vocab_size, order=order, params=min_params)
    test_loss = model.get_entropy(test_loader)

    return test_loss, val_loss


def run_language(lang, token_map, args, order=3):
    vocab_size = len(token_map)
    train_loader, val_loader, test_loader = get_data(lang)
    full_avg_len = np.mean([len(x) for x in (train_loader + val_loader + test_loader)]) - 2
    avg_len = np.mean([len(x) for x in (train_loader)]) - 2

    test_loss, val_loss = optimize_model_rnd(
        args.model, train_loader, val_loader, test_loader, vocab_size, order=order, tests=50)
    print('Test loss: %.4f  Val loss: %.4f' % (test_loss, val_loss))

    return full_avg_len, avg_len, test_loss, val_loss


def get_optimal_loss(train_loader, val_loader, test_loader, vocab_size, xp, yp, args):
    opt_params = xp[np.argmin(yp)]
    opt_params = opt_params / np.sum(opt_params, keepdims=True)

    order = len(opt_params)
    params = opt_params
    print('Best hyperparams hs: %d, params: %s' % (order, params))

    val_loss = get_model_entropy(args.model, train_loader, val_loader, vocab_size, params=params, order=order)
    test_loss = get_model_entropy(args.model, train_loader, test_loader, vocab_size, params=params, order=order)

    return test_loss, val_loss, opt_params


def _run_language_bayesian(lang, train_loader, val_loader, test_loader, token_map, args, max_order=3):
    vocab_size = len(token_map)

    full_avg_len = np.mean([len(x) for x in (train_loader)] +
                           [len(x) for x in (val_loader)] +
                           [len(x) for x in (test_loader)]) - 2
    avg_len = np.mean([len(x) for x in (train_loader)]) - 2

    n_iters = 45
    unigram_min_max = [[0.1, 1.]]
    ngram_min_max = [[0.0, 1.]] * (max_order - 1)
    bounds = np.array(unigram_min_max + ngram_min_max)
    n_pre_samples = 5

    sample_loss = sample_loss_getter(lang, train_loader, val_loader, test_loader, token_map, args)
    xp, yp = bayesian_optimisation(n_iters, sample_loss, bounds, n_pre_samples=n_pre_samples)

    test_loss, val_loss, opt_params = get_optimal_loss(train_loader, val_loader, test_loader, vocab_size, xp, yp, args)
    print('Test loss: %.4f  Val loss: %.4f' % (test_loss, val_loss))

    return full_avg_len, avg_len, test_loss, val_loss, opt_params, xp, yp


def run_language_bayesian(lang, token_map, args, max_order=3):
    train_loader, val_loader, test_loader = get_data(lang)
    full_avg_len, avg_len, test_loss, \
        val_loss, opt_params, _, _ = _run_language_bayesian(
            lang, train_loader, val_loader, test_loader, token_map, args, max_order=max_order)
    return full_avg_len, avg_len, test_loss, val_loss, opt_params


def run_languages(args):
    print('------------------- Start -------------------')
    languages, token_map, data_split, _, _ = read_info()
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    max_order = 3
    results = [['lang', 'full_avg_len', 'avg_len', 'test_loss', 'val_loss'] +
               ['param_%d' % i for i in range(max_order)]]
    for i, lang in enumerate(languages):
        print()
        print('%d Language %s' % (i, lang))
        full_avg_len, avg_len, test_loss, val_loss, opt_params = \
            run_language_bayesian(lang, token_map, args, max_order=max_order)
        results += [[lang, full_avg_len, avg_len, test_loss, val_loss] + opt_params.tolist()]

        write_csv(results, '%s/ngram.csv' % (args.rfolder))

    write_csv(results, '%s/ngram-final.csv' % (args.rfolder))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='normal')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    run_languages(args)
