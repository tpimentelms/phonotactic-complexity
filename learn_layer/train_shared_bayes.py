import numpy as np

from gp import bayesian_optimisation
from train_shared import get_model_entropy, read_info, get_data_loaders, write_csv, run_language
from util import argparser


results = [['lang', 'embedding_size', 'hidden_size', 'nlayers', 'dropout',
            'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
wait_epochs = 10


def sample_loss_getter(languages, token_map, concept_ids, args):
    global count
    train_loaders, val_loaders, test_loaders = get_data_loaders(languages, batch_size=32)
    count = 0

    def sample_loss(hyper_params):
        global results, count
        count += 1

        # hidden_size, nlayers, dropout = hyper_params
        embedding_size = int(hyper_params[0])
        hidden_size = int(hyper_params[1])
        nlayers = int(max(1, hyper_params[2]))
        dropout = max(0, hyper_params[3])
        print('%d: emb-hs %d  hs %d  nlayers %d  drop %.3f' % (count, embedding_size, hidden_size, nlayers, dropout))

        test_results, test_loss, test_acc, best_epoch, val_loss, val_acc = get_model_entropy(
            args.model, languages, train_loaders, val_loaders, test_loaders,
            token_map, concept_ids, embedding_size, hidden_size, nlayers, dropout, args,
            wait_epochs=wait_epochs, per_word=False)

        results += [['full', embedding_size, hidden_size, nlayers, dropout,
                     test_loss, test_acc, best_epoch, val_loss, val_acc]]
        for lang, result in test_results.items():
            results += [[lang, embedding_size, hidden_size, nlayers, dropout] + list(result)]

        write_csv(results, '%s/%s__bayesian-shared-full-results.csv' % (args.rfolder, args.model))
        return val_loss

    return sample_loss


def get_optimal_loss(languages, token_map, xp, yp, concept_ids, args):
    best_hyperparams = xp[np.argmin(yp)]
    embedding_size = int(best_hyperparams[0])
    hidden_size = int(best_hyperparams[1])
    nlayers = int(max(1, best_hyperparams[2]))
    dropout = max(0, best_hyperparams[3])
    print('Best hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    test_results, test_loss, test_acc, best_epoch, val_loss, val_acc = run_language(
        languages, token_map, concept_ids, args, embedding_size=embedding_size,
        hidden_size=hidden_size, nlayers=nlayers, dropout=dropout)
    return ['full', test_loss, test_acc, best_epoch, val_loss, val_acc,
            embedding_size, hidden_size, nlayers, dropout], test_results


def optimize_languages(args):
    print('------------------- Start -------------------')
    languages, token_map, data_split, concept_ids = read_info()
    print('Model %s' % args.model)
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    n_iters = 45
    bounds = np.array([[4, 256], [32, 256], [1, 2.95], [0.0, 0.5]])
    n_pre_samples = 5

    sample_loss = sample_loss_getter(languages, token_map, concept_ids, args)
    xp, yp = bayesian_optimisation(n_iters, sample_loss, bounds, n_pre_samples=n_pre_samples)

    opt_results, test_results = get_optimal_loss(languages, token_map, xp, yp, concept_ids, args)

    log_results = [['lang', 'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc',
                    'embedding_size', 'hidden_size', 'nlayers', 'dropout']]
    log_results += [opt_results]
    log_results += [[]]
    for lang, result in test_results.items():
        log_results += [[lang] + list(result)]

    write_csv(log_results, '%s/%s__bayesian-shared-results.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='bayes-opt')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    optimize_languages(args)
