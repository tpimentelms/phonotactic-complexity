import sys
sys.path.append('./')

from train_base import write_csv, read_info
from train_artificial import get_languages, add_new_symbols_to_vocab, fill_artificial_args, \
    read_artificial_data, split_data, get_phones_info, filter_test_split
from train_artificial_cv import get_data_split_cv
from train_ngram import _run_language_bayesian, remove_pads
from train_ngram_cv import _run_language_opt
from util import argparser

full_results = [['lang', 'artificial', 'fold', 'full_avg_len', 'avg_len', 'test_loss', 'val_loss']]


def get_data(lang, token_map, args, artificial=True):
    _, _, data_split, _, _ = read_info()
    return _get_data(data_split, lang, token_map, args, artificial=artificial)


def get_data_cv(fold, nfolds, lang, token_map, args, artificial=True):
    data_split = get_data_split_cv(fold, nfolds)
    return _get_data(data_split, lang, token_map, args, artificial=artificial)


def _get_data(data_split, lang, token_map, args, artificial=True):
    df = read_artificial_data(args.ffolder, lang, args.is_devoicing)
    test_split = filter_test_split(df, data_split[2])

    train_loader = get_data_split(df, data_split[0], token_map, 'train', args, artificial=artificial)
    val_loader = get_data_split(df, data_split[1], token_map, 'val', args, artificial=artificial)
    test_loader = get_data_split(df, test_split, token_map, 'test', args, artificial=artificial)
    get_phones_info(df, artificial, args)

    return train_loader, val_loader, test_loader


def get_data_split(df, concepts, token_map, mode, args, artificial=True):
    data = split_data(df, concepts, token_map, mode, args, artificial=artificial)
    return remove_pads(data, mode)


def run_artificial_language_cv(lang, token_map, args, artificial=True, max_order=3):
    global full_results, fold
    nfolds = 10
    avg_test_loss, avg_val_loss = 0, 0

    train_loader, val_loader, test_loader = get_data(lang, token_map, args, artificial=artificial)
    full_avg_len, avg_len, _, _, _, xp, yp = _run_language_bayesian(
        lang, train_loader, val_loader, test_loader, token_map,
        args, max_order=max_order)

    for fold in range(nfolds):
        print()
        print('Fold:', fold, end=' ')
        train_loader, val_loader, test_loader = get_data_cv(
            fold, nfolds, lang, token_map, args, artificial=artificial)

        full_avg_len_tmp, avg_len_tmp, test_loss, val_loss, opt_params = _run_language_opt(
            lang, train_loader, val_loader, test_loader, token_map, xp, yp,
            args, max_order=max_order)

        full_results += [[lang, artificial, fold, full_avg_len_tmp, avg_len_tmp,
                          test_loss, val_loss]]  # + opt_params.tolist()]

        avg_test_loss += test_loss / nfolds
        avg_val_loss += val_loss / nfolds

        write_csv(full_results, '%s/artificial__%s__full-results.csv' % (args.rfolder, args.model))

    return full_avg_len, avg_len, avg_test_loss, avg_val_loss


def run_languages_cv(args):
    print('------------------- Start -------------------')
    _, token_map, data_split, _, _ = read_info()
    languages = get_languages(is_devoicing=args.is_devoicing)
    token_map = add_new_symbols_to_vocab(token_map)
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    max_order = 3
    results = [['lang', 'artificial', 'full_avg_len', 'avg_len', 'test_loss', 'val_loss']]
    for i, lang in enumerate(languages):
        for artificial in [True, False]:
            print()
            print(i, end=' ')
            full_avg_len, avg_len, test_loss, val_loss = run_artificial_language_cv(
                lang, token_map, args, artificial=artificial, max_order=max_order)
            results += [[lang, artificial, full_avg_len, avg_len, test_loss, val_loss]]

            write_csv(results, '%s/artificial__%s__results.csv' % (args.rfolder, args.model))
    write_csv(results, '%s/artificial__%s__results-final.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='artificial/%s/cv')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    fill_artificial_args(args)
    run_languages_cv(args)
