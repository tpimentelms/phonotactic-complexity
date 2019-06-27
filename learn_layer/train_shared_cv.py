import sys
sys.path.append('./')
from model import opt_params
from util import argparser
from train_shared import read_info, get_model_entropy, write_csv
from train_base_cv import get_data_loaders_cv as get_lang_loader_cv

full_results = [['lang', 'fold', 'embedding_size', 'hidden_size', 'nlayers', 'dropout',
                 'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]


def get_data_loaders_cv(ffolder, fold, nfolds, languages, token_map, concept_ids):
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    verbose = True
    for lang in languages:
        train_loaders[lang], val_loaders[lang], test_loaders[lang] = \
            get_lang_loader_cv(ffolder, fold, nfolds, lang, token_map, concept_ids, verbose=verbose)
        verbose = False

    return train_loaders, val_loaders, test_loaders


def run_language_cv(languages, token_map, concept_ids, args, embedding_size=None,
                    hidden_size=256, nlayers=1, dropout=0.2):
    global full_results, fold
    nfolds = 10
    avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc = 0, 0, 0, 0
    avg_results = {lang: [0] * 2 for lang in languages}
    for fold in range(nfolds):
        print()
        print('Fold:', fold)
        train_loaders, val_loaders, test_loaders = get_data_loaders_cv(
            args.ffolder, fold, nfolds, languages, token_map, concept_ids)

        results, test_loss, test_acc, best_epoch, val_loss, val_acc = get_model_entropy(
            args.model, languages, train_loaders, val_loaders, test_loaders, token_map,
            concept_ids, embedding_size, hidden_size, nlayers, dropout, args, per_word=False)

        full_results += [['full', fold, embedding_size, hidden_size, nlayers, dropout,
                          test_loss, test_acc, best_epoch, val_loss, val_acc]]
        for lang, result in results.items():
            full_results += [[lang, fold, embedding_size, hidden_size, nlayers, dropout] + list(result)]
            avg_results[lang] = [x + (y / nfolds) for x, y in zip(avg_results[lang], result)]

        avg_test_loss += test_loss / nfolds
        avg_test_acc += test_acc / nfolds
        avg_val_loss += val_loss / nfolds
        avg_val_acc += val_acc / nfolds

        write_csv(full_results, '%s/%s__full-results.csv' % (args.rfolder, args.model))

    return avg_results, avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc


def run_opt_language_cv(languages, token_map, concept_ids, args):
    embedding_size, hidden_size, nlayers, dropout = opt_params.get_shared_opt_params(args.model)
    print('Optimum hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    return run_language_cv(languages, token_map, concept_ids, args,
                           embedding_size=embedding_size, hidden_size=hidden_size,
                           nlayers=nlayers, dropout=dropout)


def run_languages(args):
    print('------------------- Start -------------------')
    languages, token_map, data_split, concept_ids = read_info()
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    if args.opt:
        test_results, test_loss, \
            test_acc, val_loss, val_acc = run_opt_language_cv(languages, token_map, concept_ids, args)
    else:
        test_results, test_loss, \
            test_acc, val_loss, val_acc = run_language_cv(languages, token_map, concept_ids, args)

    results = [['lang', 'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
    results += [['full', test_loss, test_acc, '-', val_loss, val_acc]]
    for lang, result in test_results.items():
        results += [[lang] + list(result)]

    write_csv(results, '%s/%s__shared-results.csv' % (args.rfolder, args.model))
    write_csv(results, '%s/%s__shared-results-final.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='cv')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    run_languages(args)
