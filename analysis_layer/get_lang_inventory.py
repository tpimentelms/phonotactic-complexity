import pandas as pd
import numpy as np

import sys
sys.path.append('./')
from data_layer.phoible import PhoibleInfo
from data_layer.parse import read_src_data, get_languages, separate_train, separate_per_language
from util import argparser


def get_symbols(df, field='IPA'):
    symbols = set([])
    for i, (index, x) in enumerate(df.iterrows()):
        symbols |= set([y for y in x[field].split(' ')])

    return symbols


def get_lang_len(df, field='IPA'):
    lens = []
    for i, (index, x) in enumerate(df.iterrows()):
        word = x[field].split(' ')
        lens += [len(word)]

    return np.mean(lens)


def get_lang_ipa_info(df, languages_df, args, field='IPA'):
    phoible = PhoibleInfo()
    lang_data = []

    for lang, lang_df in languages_df.items():
        frames = [lang_df['train'], lang_df['val'], lang_df['test']]
        full_data = pd.concat(frames)

        avg_len = get_lang_len(full_data, field=field)
        symbols = get_symbols(full_data, field=field)
        consonant, vowel, tone, symbol, unrecognized = phoible.count_types(symbols)

        lang_data += [[lang, len(symbols), vowel, consonant, tone, unrecognized, avg_len]]

    columns = ['lang', 'inventory', 'vowel', 'consonant', 'tone', 'unrecognized', 'avg_len']
    df_info = pd.DataFrame(lang_data, columns=columns)
    rfolder = args.rfolder[:-len('orig')]
    df_info.to_csv('%s/lang_inventory.csv' % (rfolder))


def main(args):
    df = read_src_data(args.ffolder)

    languages = get_languages(df)
    train_df, val_df, test_df, _ = separate_train(df)
    languages_df = separate_per_language(train_df, val_df, test_df, languages)

    get_lang_ipa_info(df, languages_df, args, field='IPA')


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='inventory')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    main(args)
