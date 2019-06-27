import pandas as pd
import numpy as np

import sys
sys.path.append('./')
from train_base import write_csv, read_info, convert_to_loader, _run_language
from util import argparser

full_results = [['lang', 'artificial', 'avg_len', 'test_shannon', 'test_loss',
                 'test_acc', 'val_loss', 'val_acc', 'best_epoch']]


def get_data_loaders(ffolder, lang, is_devoicing, token_map, args, artificial=True):
    _, _, data_split, _, _ = read_info()
    return _get_data_loaders(data_split, ffolder, lang, is_devoicing, token_map, args, artificial=artificial)


def _get_data_loaders(data_split, ffolder, lang, is_devoicing, token_map, args, artificial=True):
    df = read_artificial_data(ffolder, lang, is_devoicing)
    test_split = filter_test_split(df, data_split[2])

    train_loader = get_data_loader(df, data_split[0], token_map, 'train', args, artificial=artificial)
    val_loader = get_data_loader(df, data_split[1], token_map, 'val', args, artificial=artificial)
    test_loader = get_data_loader(df, test_split, token_map, 'test', args, artificial=artificial)
    get_phones_info(df, artificial, args)

    return train_loader, val_loader, test_loader


def filter_test_split(df, raw_test_split):
    df = df[df[0].str.match('.*::N$')]
    concepts = list(df[0].unique())
    test_split = [x for x in raw_test_split if x in concepts]
    print('Test %d' % (len(test_split)))
    return test_split


def get_data_loader(df, concepts, token_map, mode, args, artificial=True):
    data = split_data(df, concepts, token_map, mode, args, artificial=artificial)
    return convert_to_loader(data, mode)


def get_phones_info(df, artificial, args):
    col = args.col_artificial if artificial else args.col_normal
    phones = set([y for x in df[col].values for y in x])
    print('Phones %d' % len(phones))


def read_artificial_data(ffolder, lang, is_devoicing):
    artificial_folder = 'devoicing' if is_devoicing else 'harmony'
    return pd.read_csv('%s/artificial/%s/%s2' % (ffolder, artificial_folder, lang), delimiter='\t', header=None)


def split_data(df, concepts, token_map, mode, args, artificial=True):
    col = args.col_artificial if artificial else args.col_normal
    df_partial = df[df[0].isin(concepts)]
    data_partial = df_partial[col].values
    print('Differences %s: %d\tNo difference: %d' %
          (mode, (df_partial[args.col_normal] != df_partial[args.col_artificial]).sum(),
           (df_partial[args.col_normal] == df_partial[args.col_artificial]).sum()))

    max_len = max([len(x) for x in data_partial])
    data = np.zeros((len(data_partial), max_len + 2))
    data.fill(token_map['PAD'])
    for i, string in enumerate(data_partial):
        _data = [token_map['SOW']] + [token_map[x] for x in string.split(' ')] + [token_map['EOW']]
        data[i, :len(_data)] = _data

    return data


def run_artificial_language(lang, is_devoicing, token_map, concept_ids, ipa_to_concepts, args, artificial=True,
                            embedding_size=None, hidden_size=256, nlayers=1, dropout=0.2):
    train_loader, val_loader, test_loader = get_data_loaders(
        args.ffolder, lang, is_devoicing, token_map, args, artificial=artificial)
    return _run_language(
        '%s %s' % (lang, 'art' if artificial else 'norm'),
        train_loader, val_loader, test_loader, token_map, ipa_to_concepts,
        args, embedding_size=embedding_size, hidden_size=hidden_size, nlayers=nlayers, dropout=dropout)


def get_languages(is_devoicing=True):
    devoicing_langs = ['deu', 'nld']
    harmony_langs = ['bua', 'ckt', 'evn', 'fin', 'hun', 'khk', 'mhr', 'mnc', 'myv', 'tel', 'tur']
    return devoicing_langs if is_devoicing else harmony_langs


def add_new_symbols_to_vocab(token_map):
    new_symbols = ['g', 'ʉʲ', 'uʲ', 'ɹʲʲ', 'iʲʲ', 'ɵʲ', 'õ̃', 'ʋ̃', 'ũ̃', 'ĩ̃', 'ʌ̃̃', 'ẽ̃']
    for symb in new_symbols:
        token_map[symb] = max(token_map.values()) + 1
    return token_map


def fill_artificial_args(args):
    args.is_devoicing = (args.artificial_type == 'devoicing')
    args.col_artificial = 3 if args.is_devoicing else 2
    args.col_normal = 2 if args.is_devoicing else 3


def run_languages(args):
    print('------------------- Start -------------------')
    _, token_map, data_split, concept_ids, ipa_to_concepts = read_info()

    languages = get_languages(is_devoicing=args.is_devoicing)
    token_map = add_new_symbols_to_vocab(token_map)
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    results = [['lang', 'avg_len', 'test_shannon', 'test_loss', 'test_acc', 'val_loss', 'val_acc']]
    for i, lang in enumerate(languages):
        for artificial in [True, False]:
            print()
            print('%d. %s %s' % (i, lang, 'artificial' if artificial else 'default'))
            avg_len, shannon, test_shannon, test_loss, \
                test_acc, best_epoch, val_loss, val_acc = run_artificial_language(
                    lang, args.is_devoicing, token_map, concept_ids, ipa_to_concepts, args, artificial=artificial)
            results += [['%s %s' % (lang, 'art' if artificial else 'norm'),
                         avg_len, shannon, test_shannon, test_loss, test_acc,
                         best_epoch, val_loss, val_acc]]

            write_csv(results, '%s/artificial__%s__results.csv' % (args.rfolder, args.model))
    write_csv(results, '%s/artificial__%s__results-final.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='artificial/%s/normal')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    fill_artificial_args(args)
    run_languages(args)
