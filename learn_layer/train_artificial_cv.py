import numpy as np

import sys
sys.path.append('./')
from model import opt_params
from train_base import write_csv, read_info, _run_language
from train_artificial import get_languages, add_new_symbols_to_vocab, \
    fill_artificial_args, _get_data_loaders
from util import argparser

full_results = [['lang', 'artificial', 'fold', 'avg_len', 'test_shannon', 'test_loss',
                 'test_acc', 'val_loss', 'val_acc', 'best_epoch']]


def get_data_loaders_cv(ffolder, fold, nfolds, lang, is_devoicing, token_map, args, artificial=True):
    data_split = get_data_split_cv(fold, nfolds)
    return _get_data_loaders(data_split, ffolder, lang, is_devoicing, token_map, args, artificial=artificial)


def get_nouns_split_cv(fold, nfolds, df):
    df = df[df[0].str.match('.*::N$')]
    concepts = list(df[0].unique())
    np.random.shuffle(concepts)

    return _get_data_split_cv(fold, nfolds, concepts)


def get_data_split_cv(fold, nfolds):
    _, _, data_split, _, _ = read_info()
    concepts = [y for x in data_split for y in x]

    return _get_data_split_cv(fold, nfolds, concepts)


def _get_data_split_cv(fold, nfolds, concepts):
    part_size = int(len(concepts) / nfolds)
    test_fold = (fold + 1) % nfolds
    train_start_fold = 0 if test_fold > fold else (test_fold + 1)

    train = concepts[train_start_fold * part_size:fold * part_size] + concepts[(fold + 2) * part_size:]
    val = concepts[fold * part_size: (fold + 1) * part_size]
    test = concepts[(test_fold) * part_size:(test_fold + 1) * part_size]

    print('Train %d, Val %d, Test %d' % (len(train), len(val), len(test)))

    return (train, val, test)


def run_artificial_language_cv(lang, is_devoicing, token_map, concept_ids, ipa_to_concepts, args, artificial=True,
                               embedding_size=None, hidden_size=256, nlayers=1, dropout=0.2):
    global full_results
    nfolds = 10
    avg_shannon, avg_test_shannon, avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc = 0, 0, 0, 0, 0, 0
    for fold in range(nfolds):
        print()
        print(fold, end=' ')
        print('Best hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
              (embedding_size, hidden_size, nlayers, dropout))
        train_loader, val_loader, test_loader = \
            get_data_loaders_cv(args.ffolder, fold, nfolds, lang, is_devoicing, token_map, args, artificial=artificial)
        avg_len, shannon, test_shannon, test_loss, \
            test_acc, best_epoch, val_loss, val_acc = _run_language(
                '%s %s' % (lang, 'art' if artificial else 'norm'),
                train_loader, val_loader, test_loader, token_map,
                ipa_to_concepts, args, embedding_size=embedding_size,
                hidden_size=hidden_size, nlayers=nlayers, dropout=dropout)

        full_results += [[lang, artificial, fold, avg_len, test_shannon,
                          test_loss, test_acc, val_loss, val_acc, best_epoch]]

        avg_shannon += shannon / nfolds
        avg_test_shannon += test_shannon / nfolds
        avg_test_loss += test_loss / nfolds
        avg_test_acc += test_acc / nfolds
        avg_val_loss += val_loss / nfolds
        avg_val_acc += val_acc / nfolds

        write_csv(full_results, '%s/artificial__%s__full-results.csv' % (args.rfolder, args.model))

    return avg_len, avg_shannon, avg_test_shannon, avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc


def run_languages_cv(args):
    print('------------------- Start -------------------')
    _, token_map, data_split, concept_ids, ipa_to_concepts = read_info()
    languages = get_languages(is_devoicing=args.is_devoicing)
    token_map = add_new_symbols_to_vocab(token_map)
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    results = [['lang', 'artificial', 'avg_len', 'shannon', 'test_shannon',
                'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
    for i, lang in enumerate(languages):
        for artificial in [True, False]:
            print()
            print('%d. %s %s' % (i, lang, 'artificial' if artificial else 'default'))
            embedding_size, hidden_size, nlayers, dropout = \
                opt_params.get_artificial_opt_params(args.model, lang, artificial, args.artificial_type, args.data)
            avg_len, shannon, test_shannon, test_loss, \
                test_acc, val_loss, val_acc = run_artificial_language_cv(
                    lang, args.is_devoicing, token_map, concept_ids, ipa_to_concepts, args,
                    artificial=artificial, embedding_size=embedding_size,
                    hidden_size=hidden_size, nlayers=nlayers, dropout=dropout)
            results += [[lang, artificial, avg_len, shannon, test_shannon, test_loss, test_acc, val_loss, val_acc]]

            write_csv(results, '%s/artificial__%s__results.csv' % (args.rfolder, args.model))
    write_csv(results, '%s/artificial__%s__results-final.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='artificial/%s/cv')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    fill_artificial_args(args)
    run_languages_cv(args)
