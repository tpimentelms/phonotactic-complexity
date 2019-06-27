import numpy as np

from gp import bayesian_optimisation
from train_base import get_model_entropy, read_info, get_data_loaders, write_csv, run_language
from util import argparser


results = [['lang', 'embedding_size', 'hidden_size', 'nlayers', 'dropout',
            'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
wait_epochs = 10


def sample_loss_getter(lang, token_map, ipa_to_concept, args):
    global count
    train_loader, val_loader, test_loader = get_data_loaders(lang)
    count = 0

    def sample_loss(hyper_params):
        global results, count
        count += 1

        embedding_size = int(hyper_params[0])
        hidden_size = int(hyper_params[1])
        nlayers = int(max(1, hyper_params[2]))
        dropout = max(0, hyper_params[3])
        print('%d: emb-hs %d  hs %d  nlayers %d  drop %.3f' % (count, embedding_size, hidden_size, nlayers, dropout))

        test_loss, test_acc, best_epoch, val_loss, val_acc = get_model_entropy(
            lang, args.model, train_loader, val_loader, test_loader, token_map, ipa_to_concept,
            embedding_size, hidden_size, nlayers, dropout, args, wait_epochs=wait_epochs, per_word=False)

        results += [[lang, embedding_size, hidden_size, nlayers, dropout,
                     test_loss, test_acc, best_epoch, val_loss, val_acc]]
        return val_loss

    return sample_loss


def get_optimal_loss(lang, token_map, xp, yp, ipa_to_concept, args):
    best_hyperparams = xp[np.argmin(yp)]
    embedding_size = int(best_hyperparams[0])
    hidden_size = int(best_hyperparams[1])
    nlayers = int(max(1, best_hyperparams[2]))
    dropout = max(0, best_hyperparams[3])
    print('Best hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    avg_len, shannon, test_shannon, test_loss, \
        test_acc, best_epoch, val_loss, val_acc = run_language(
            lang, token_map, ipa_to_concept, args, embedding_size=embedding_size,
            hidden_size=hidden_size, nlayers=nlayers, dropout=dropout)
    return [lang, avg_len, shannon, test_shannon, test_loss, test_acc, best_epoch,
            val_loss, val_acc, embedding_size, hidden_size, nlayers, dropout]


def optimize_languages(args):
    languages, token_map, data_split, _, ipa_to_concept = read_info()
    print('Model %s' % args.model)
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    n_iters = 45
    bounds = np.array([[4, 256], [32, 256], [1, 2.95], [0.0, 0.5]])
    n_pre_samples = 5

    opt_results = [['lang', 'avg_len', 'shannon', 'test_shannon', 'test_loss', 'test_acc',
                    'best_epoch', 'val_loss', 'val_acc', 'embedding_size', 'hidden_size',
                    'nlayers', 'dropout']]

    for i, lang in enumerate(languages):
        print()
        print('%d. %s' % (i, lang))
        sample_loss = sample_loss_getter(lang, token_map, ipa_to_concept, args)
        xp, yp = bayesian_optimisation(n_iters, sample_loss, bounds, n_pre_samples=n_pre_samples)

        opt_results += [get_optimal_loss(lang, token_map, xp, yp, ipa_to_concept, args)]

        write_csv(results, '%s/%s__baysian-results.csv' % (args.rfolder, args.model))
        write_csv(opt_results, '%s/%s__opt-results.csv' % (args.rfolder, args.model))

    write_csv(results, '%s/%s__baysian-results-final.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='bayes-opt')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    optimize_languages(args)
