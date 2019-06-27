import numpy as np

import sys
sys.path.append('./')
from train_base import read_info, write_csv
from train_base_cv import split_data, get_data_split_cv, get_lang_df
from train_ngram import _run_language_bayesian, get_data, remove_pads, get_optimal_loss
from util import argparser

results = [['lang', 'order', 'params', 'val_loss']]
full_results = [['lang', 'fold', 'avg_len', 'test_loss', 'val_loss']]


def _run_language_opt(lang, train_loader, val_loader, test_loader, token_map, xp, yp, args, max_order=3):
    vocab_size = len(token_map)
    full_avg_len = np.mean([len(x) for x in (train_loader + val_loader + test_loader)]) - 2
    avg_len = np.mean([len(x) for x in (train_loader)]) - 2

    test_loss, val_loss, opt_params = get_optimal_loss(train_loader, val_loader, test_loader, vocab_size, xp, yp, args)
    print('Test loss: %.4f  Val loss: %.4f' % (test_loss, val_loss))

    return full_avg_len, avg_len, test_loss, val_loss, opt_params


def get_data_cv(ffolder, fold, nfolds, lang, token_map, concept_ids, verbose=True):
    global data_split
    data_split = get_data_split_cv(fold, nfolds, verbose=verbose)
    df = get_lang_df(lang, ffolder)

    train_loader = get_ngram_data(df, data_split[0], token_map, 'train', concept_ids)
    val_loader = get_ngram_data(df, data_split[1], token_map, 'val', concept_ids)
    test_loader = get_ngram_data(df, data_split[2], token_map, 'test', concept_ids)

    return train_loader, val_loader, test_loader


def get_ngram_data(df, concepts, token_map, mode, concept_ids):
    data = split_data(df, concepts, token_map, mode, concept_ids)
    data = remove_ids(data)
    return remove_pads(data, mode)


def remove_ids(data):
    return data[:, :-1]


def run_language_cv(lang, token_map, concept_ids, args, max_order=3):
    global full_results, fold
    nfolds = 10
    avg_test_loss, avg_val_loss = 0, 0

    train_loader, val_loader, test_loader = get_data(lang)
    full_avg_len, avg_len, _, _, _, xp, yp = _run_language_bayesian(
        lang, train_loader, val_loader, test_loader, token_map,
        args, max_order=max_order)

    for fold in range(nfolds):
        print()
        print('Fold:', fold, end=' ')

        train_loader, val_loader, test_loader = get_data_cv(args.ffolder, fold, nfolds, lang, token_map, concept_ids)
        _, _, test_loss, val_loss, opt_params = _run_language_opt(
            lang, train_loader, val_loader, test_loader, token_map, xp, yp,
            args, max_order=max_order)

        full_results += [[lang, fold, avg_len, test_loss, val_loss]]  # + opt_params.tolist()]

        avg_test_loss += test_loss / nfolds
        avg_val_loss += val_loss / nfolds

        write_csv(full_results, '%s/%s__full-results.csv' % (args.rfolder, args.model))

    return full_avg_len, avg_len, avg_test_loss, avg_val_loss


def run_languages(args):
    print('------------------- Start -------------------')
    languages, token_map, data_split, concept_ids, _ = read_info()
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    max_order = 3
    results = [['lang', 'full_avg_len', 'avg_len', 'test_loss', 'val_loss'] +
               ['param_%d' % i for i in range(max_order)]]
    for i, lang in enumerate(languages):
        print()
        print('%d Language %s' % (i, lang))
        full_avg_len, avg_len, test_loss, val_loss = \
            run_language_cv(lang, token_map, concept_ids, args, max_order=max_order)
        results += [[lang, full_avg_len, avg_len, test_loss, val_loss]]  # + opt_params.tolist()]

        write_csv(results, '%s/ngram.csv' % (args.rfolder))

    write_csv(results, '%s/ngram-final.csv' % (args.rfolder))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='cv')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    run_languages(args)
