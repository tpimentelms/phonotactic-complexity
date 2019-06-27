import numpy as np

import sys
sys.path.append('./')
from data_layer.parse import read_src_data
from model import opt_params
from util import argparser
from train_base import read_info, write_csv, convert_to_loader, _run_language

full_results = [['lang', 'fold', 'avg_len', 'test_shannon', 'test_loss',
                 'test_acc', 'val_loss', 'val_acc', 'best_epoch']]


def get_lang_df(lang, ffolder):
    df = read_src_data(ffolder)
    return df[df['Language_ID'] == lang]


def get_data_loaders_cv(ffolder, fold, nfolds, lang, token_map, concept_ids, verbose=True):
    global data_split
    data_split = get_data_split_cv(fold, nfolds, verbose=verbose)
    df = get_lang_df(lang, ffolder)

    train_loader = get_data_loader(df, data_split[0], token_map, 'train', concept_ids)
    val_loader = get_data_loader(df, data_split[1], token_map, 'val', concept_ids)
    test_loader = get_data_loader(df, data_split[2], token_map, 'test', concept_ids)

    return train_loader, val_loader, test_loader


def get_data_split_cv(fold, nfolds, verbose=True):
    _, _, data_split, _, _ = read_info()
    concepts = [y for x in data_split for y in x]

    return _get_data_split_cv(fold, nfolds, concepts, verbose=verbose)


def _get_data_split_cv(fold, nfolds, concepts, verbose=True):
    part_size = int(len(concepts) / nfolds)
    test_fold = (fold + 1) % nfolds
    train_start_fold = 0 if test_fold > fold else (test_fold + 1)

    train = concepts[train_start_fold * part_size:fold * part_size]
    train += concepts[(fold + 2) * part_size:] if fold + 2 < nfolds else []
    val = concepts[fold * part_size:(fold + 1) * part_size] if fold + 1 < nfolds else concepts[fold * part_size:]
    test = concepts[(test_fold) * part_size:(test_fold + 1) * part_size] if test_fold + 1 < nfolds \
        else concepts[(test_fold) * part_size:]

    if verbose:
        print('Train %d, Val %d, Test %d' % (len(train), len(val), len(test)))

    return (train, val, test)


def get_data_loader(df, concepts, token_map, mode, concept_ids):
    data = split_data(df, concepts, token_map, mode, concept_ids)
    return convert_to_loader(data, mode)


def split_data(df, concepts, token_map, mode, concept_ids):
    df_partial = df[df['Concept_ID'].isin(set(concepts))]
    data_partial = df_partial['IPA'].values
    ids = df_partial.index

    max_len = max([len(x) for x in data_partial])
    data = np.zeros((len(data_partial), max_len + 3))
    data.fill(token_map['PAD'])
    for i, (string, _id) in enumerate(zip(data_partial, ids)):
        instance = string.split(' ')
        _data = [token_map['SOW']] + [token_map[x] for x in instance] + [token_map['EOW']]
        data[i, :len(_data)] = _data
        data[i, -1] = _id

    return data


def run_language_cv(lang, token_map, concept_ids, ipa_to_concept, args, embedding_size=None,
                    hidden_size=256, nlayers=1, dropout=0.2):
    global full_results, fold
    nfolds = 10
    avg_shannon, avg_test_shannon, avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc = 0, 0, 0, 0, 0, 0
    for fold in range(nfolds):
        print()
        print('Fold:', fold, end=' ')

        train_loader, val_loader, test_loader = get_data_loaders_cv(
            args.ffolder, fold, nfolds, lang, token_map, concept_ids)
        avg_len, shannon, test_shannon, test_loss, \
            test_acc, best_epoch, val_loss, val_acc = _run_language(
                lang, train_loader, val_loader, test_loader, token_map, ipa_to_concept,
                args, embedding_size=embedding_size, hidden_size=hidden_size,
                nlayers=nlayers, dropout=dropout, per_word=True)

        full_results += [[lang, fold, avg_len, test_shannon, test_loss, test_acc, val_loss, val_acc, best_epoch]]

        avg_shannon += shannon / nfolds
        avg_test_shannon += test_shannon / nfolds
        avg_test_loss += test_loss / nfolds
        avg_test_acc += test_acc / nfolds
        avg_val_loss += val_loss / nfolds
        avg_val_acc += val_acc / nfolds

        write_csv(full_results, '%s/%s__full-results.csv' % (args.rfolder, args.model))

    return avg_len, avg_shannon, avg_test_shannon, avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc


def run_opt_language_cv(lang, token_map, concept_ids, ipa_to_concept, args):
    embedding_size, hidden_size, nlayers, dropout = opt_params.get_opt_params(args.model, lang)
    print('Optimum hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    return run_language_cv(lang, token_map, concept_ids, ipa_to_concept, args,
                           embedding_size=embedding_size, hidden_size=hidden_size,
                           nlayers=nlayers, dropout=dropout)


def run_languages(args):
    languages, token_map, data_split, concept_ids, ipa_to_concept = read_info()
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    results = [['lang', 'avg_len', 'shannon', 'test_shannon', 'test_loss', 'test_acc', 'val_loss', 'val_acc']]
    for i, lang in enumerate(languages):
        print()
        print('Lang:', i, end=' ')

        if args.opt:
            avg_len, shannon, test_shannon, test_loss, \
                test_acc, val_loss, val_acc = run_opt_language_cv(lang, token_map, concept_ids, ipa_to_concept, args)
        else:
            avg_len, shannon, test_shannon, test_loss, \
                test_acc, val_loss, val_acc = run_language_cv(lang, token_map, concept_ids, ipa_to_concept, args)
        results += [[lang, avg_len, shannon, test_shannon, test_loss, test_acc, val_loss, val_acc]]

        write_csv(results, '%s/%s__results.csv' % (args.rfolder, args.model))
    write_csv(results, '%s/%s__results-final.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='cv')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    run_languages(args)
