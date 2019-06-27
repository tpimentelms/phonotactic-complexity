# Note: Dataset 0 is PAD 1 is SOW and 2 is EOW
import pandas as pd
import numpy as np
import pickle

import sys
sys.path.append('./')
import util.argparser as parser
from util import util


def read_src_data(ffolder):
    filename = '%s/orig.csv' % ffolder
    df = pd.read_csv(filename, sep='\t')
    del df['Glottocode']
    df = df[df.Language_ID != 'cmn']
    df = df.dropna()
    return df


def get_languages(df):
    return df.Language_ID.unique()


def get_phrases(df):
    phrases = df.Concept_ID.unique()
    np.random.shuffle(phrases)
    return phrases


def separate_df(df, train_set, val_set, test_set):
    train_df = df[df['Concept_ID'].isin(train_set)]
    val_df = df[df['Concept_ID'].isin(val_set)]
    test_df = df[df['Concept_ID'].isin(test_set)]

    return train_df, val_df, test_df


def separate_train(df):
    phrases = get_phrases(df)

    num_sentences = phrases.size

    train_size = int(num_sentences * .8)
    val_size = int(num_sentences * .1)
    test_size = num_sentences - train_size - val_size

    train_set = phrases[:train_size]
    val_set = phrases[train_size:-test_size]
    test_set = phrases[-test_size:]
    data_split = (train_set, val_set, test_set)

    train_df, val_df, test_df = separate_df(df, train_set, val_set, test_set)
    return train_df, val_df, test_df, data_split


def separate_per_language(train_df, val_df, test_df, languages):
    languages_df_train = separate_per_language_single_df(train_df, languages)
    languages_df_val = separate_per_language_single_df(val_df, languages)
    languages_df_test = separate_per_language_single_df(test_df, languages)

    languages_df = {
        lang: {
            'train': languages_df_train[lang],
            'val': languages_df_val[lang],
            'test': languages_df_test[lang],
        } for lang in languages_df_train.keys()}
    return languages_df


def separate_per_language_single_df(df, languages):
    languages_df = {lang: df[df['Language_ID'] == lang] for lang in languages}
    return languages_df


def get_tokens(df):
    tokens = set()
    for index, x in df.iterrows():
        try:
            tokens |= set(x.IPA.split(' '))
        except:
            continue

    tokens = sorted(list(tokens))
    token_map = {x: i + 3 for i, x in enumerate(tokens)}
    token_map['PAD'] = 0
    token_map['SOW'] = 1
    token_map['EOW'] = 2

    return token_map


def get_concept_ids(df):
    concepts = df.Concept_ID.unique()
    concept_ids = {x: i for i, x in enumerate(concepts)}
    strings_concepts = pd.Series(df.Concept_ID.values, index=df.index).to_dict()
    IPA_to_concept = {k: concept_ids[x] for k, x in strings_concepts.items()}

    return concept_ids, IPA_to_concept


def process_languages(languages_df, token_map, args):
    util.mkdir('%s/preprocess/' % args.ffolder)
    for lang, df in languages_df.items():
        process_language(df, token_map, lang, args)


def process_language(dfs, token_map, lang, args):
    for mode in ['train', 'val', 'test']:
        process_language_mode(dfs[mode], token_map, lang, mode, args)


def process_language_mode(df, token_map, lang, mode, args):
    data = parse_data(df, token_map)
    save_data(data, lang, mode, args.ffolder)


def parse_data(df, token_map):
    max_len = df.IPA.map(lambda x: len(x.split(' '))).max()
    data = np.zeros((df.shape[0], max_len + 3))

    for i, (index, x) in enumerate(df.iterrows()):
        try:
            instance = x.IPA.split(' ')
            data[i, 0] = 1
            data[i, 1:len(instance) + 1] = [token_map[z] for z in instance]
            data[i, len(instance) + 1] = 2
            data[i, -1] = index
        except:
            continue

    return data


def save_data(data, lang, mode, ffolder):
    with open('%s/preprocess/data-%s-%s.npy' % (ffolder, lang, mode), 'wb') as f:
        np.save(f, data)


def save_info(ffolder, languages, token_map, data_split, concepts_ids, IPA_to_concept):
    info = {
        'languages': languages,
        'token_map': token_map,
        'data_split': data_split,
        'concepts_ids': concepts_ids,
        'IPA_to_concept': IPA_to_concept,
    }
    with open('%s/preprocess/info.pckl' % ffolder, 'wb') as f:
        pickle.dump(info, f)


def load_info(args):
    with open('%s/preprocess/info.pckl' % args.ffolder, 'rb') as f:
        info = pickle.load(f)
    languages = info['languages']
    token_map = info['token_map']
    data_split = info['data_split']
    concept_ids = info['concepts_ids']

    return languages, token_map, data_split, concept_ids


def main(args):
    df = read_src_data(args.ffolder)

    languages = get_languages(df)
    train_df, val_df, test_df, data_split = separate_train(df)
    token_map = get_tokens(df)
    concepts_ids, IPA_to_concept = get_concept_ids(df)

    languages_df = separate_per_language(train_df, val_df, test_df, languages)

    process_languages(languages_df, token_map, args)
    save_info(args.ffolder, languages, token_map, data_split, concepts_ids, IPA_to_concept)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    main(args)
