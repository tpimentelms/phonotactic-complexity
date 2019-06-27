import numpy as np
from nltk.model import build_vocabulary, count_ngrams

import sys
sys.path.append('./')
from model.ngram import LaplaceUnigramModel
from train_base import read_info, write_csv
from train_ngram import get_data
from train_ngram_cv import get_data_cv
from util import argparser


results = [['lang', 'order', 'params', 'val_loss']]
full_results = [['lang', 'fold', 'full_avg_len', 'avg_len', 'test_loss', 'val_loss']]


def get_model_entropy(model, train_loader, eval_loader, vocab_size, params=None, order=1):
    if model == 'unigram':
        pass
    else:
        raise ValueError("Model not implemented: %s" % model)

    params = params / np.sum(params, keepdims=True) if params else None
    vocab = build_vocabulary(1, *train_loader)
    counter = count_ngrams(1, vocab, train_loader, pad_left=False, pad_right=False)
    model = LaplaceUnigramModel(vocab_size, counter)
    val_loss = model.get_entropy(eval_loader)

    return val_loss


def get_loss(train_loader, val_loader, test_loader, vocab_size, args):
    val_loss = get_model_entropy(args.model, train_loader, val_loader, vocab_size, order=1)
    test_loss = get_model_entropy(args.model, train_loader, test_loader, vocab_size, order=1)

    return test_loss, val_loss


def _run_language(lang, train_loader, val_loader, test_loader, token_map, args):
    vocab_size = len(token_map)
    full_avg_len = np.mean([len(x) for x in (train_loader)] +
                           [len(x) for x in (val_loader)] +
                           [len(x) for x in (test_loader)]) - 2
    avg_len = np.mean([len(x) for x in (train_loader)]) - 2

    test_loss, val_loss = get_loss(train_loader, val_loader, test_loader, vocab_size, args)
    print('Test loss: %.4f  Val loss: %.4f' % (test_loss, val_loss))

    return full_avg_len, avg_len, test_loss, val_loss


def run_language_cv(lang, token_map, concept_ids, args):
    global full_results, fold
    nfolds = 10
    avg_test_loss, avg_val_loss = 0, 0

    train_loader, val_loader, test_loader = get_data(lang)
    full_avg_len, avg_len, _, _ = _run_language(
        lang, train_loader, val_loader, test_loader, token_map,
        args)

    for fold in range(nfolds):
        print()
        print('Fold:', fold, end=' ')
        train_loader, val_loader, test_loader = get_data_cv(args.ffolder, fold, nfolds, lang, token_map, concept_ids)
        _, _, test_loss, val_loss = _run_language(
            lang, train_loader, val_loader, test_loader, token_map,
            args)

        full_results += [[lang, fold, full_avg_len, avg_len, test_loss, val_loss]]  # + opt_params.tolist()]

        avg_test_loss += test_loss / nfolds
        avg_val_loss += val_loss / nfolds

        write_csv(full_results, '%s/%s__full-results.csv' % (args.rfolder, args.model))

    return full_avg_len, avg_len, avg_test_loss, avg_val_loss


def run_languages(args):
    print('------------------- Start -------------------')
    languages, token_map, data_split, concept_ids, _ = read_info()
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    results = [['lang', 'full_avg_len', 'avg_len', 'test_loss', 'val_loss']]
    for i, lang in enumerate(languages):
        print()
        print('%d Language %s' % (i, lang))
        full_avg_len, avg_len, test_loss, val_loss = run_language_cv(lang, token_map, concept_ids, args)
        results += [[lang, full_avg_len, avg_len, test_loss, val_loss]]

        write_csv(results, '%s/unigram.csv' % (args.rfolder))

    write_csv(results, '%s/unigram-final.csv' % (args.rfolder))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='cv')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    run_languages(args)
