import numpy as np

from gp import bayesian_optimisation
from train_artificial import get_data_loaders, run_artificial_language, \
    get_languages, add_new_symbols_to_vocab, fill_artificial_args
from train_base import get_model_entropy, read_info, write_csv
from util import argparser


results = [['lang', 'artificial', 'embedding_size', 'hidden_size', 'nlayers', 'dropout',
            'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
wait_epochs = 10


def sample_loss_getter(lang, is_devoicing, token_map, ipa_to_concepts, args, artificial=True):
    global count
    train_loader, val_loader, test_loader = get_data_loaders(
        args.ffolder, lang, is_devoicing, token_map, args, artificial=artificial)
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
            lang, args.model, train_loader, val_loader, test_loader, token_map, ipa_to_concepts,
            embedding_size, hidden_size, nlayers, dropout, args, wait_epochs=wait_epochs, per_word=False)
        results += [[lang, artificial, embedding_size, hidden_size, nlayers, dropout,
                     test_loss, test_acc, best_epoch, val_loss, val_acc]]
        return val_loss

    return sample_loss


def get_optimal_loss(lang, is_devoicing, artificial, token_map, concept_ids, ipa_to_concepts, xp, yp, args):
    best_hyperparams = xp[np.argmin(yp)]
    embedding_size = int(best_hyperparams[0])
    hidden_size = int(best_hyperparams[1])
    nlayers = int(max(1, best_hyperparams[2]))
    dropout = max(0, best_hyperparams[3])
    print('Best hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    avg_len, shannon, test_shannon, test_loss, \
        test_acc, best_epoch, val_loss, val_acc = run_artificial_language(
            lang, is_devoicing, token_map, concept_ids, ipa_to_concepts, args,
            artificial=artificial, embedding_size=embedding_size,
            hidden_size=hidden_size, nlayers=nlayers, dropout=dropout)
    return [lang, artificial, avg_len, shannon, test_shannon, test_loss, test_acc, best_epoch,
            val_loss, val_acc, embedding_size, hidden_size, nlayers, dropout]


def optimize_languages(args):
    print('------------------- Start -------------------')
    _, token_map, data_split, concept_ids, ipa_to_concepts = read_info()
    languages = get_languages(is_devoicing=args.is_devoicing)
    token_map = add_new_symbols_to_vocab(token_map)
    print('Model %s' % args.model)
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    n_iters = 45
    bounds = np.array([[4, 256], [32, 256], [1, 2.95], [0.0, 0.5]])
    n_pre_samples = 5

    opt_results = [['lang', 'artificial', 'avg_len', 'shannon', 'test_shannon', 'test_loss', 'test_acc', 'best_epoch',
                    'val_loss', 'val_acc', 'embedding_size', 'hidden_size', 'nlayers', 'dropout']]
    for i, lang in enumerate(languages):
        for artificial in [True, False]:
            print()
            print('%d. %s %s' % (i, lang, 'artificial' if artificial else 'default'))

            sample_loss = sample_loss_getter(lang, args.is_devoicing, token_map, ipa_to_concepts,
                                             args, artificial=artificial)
            xp, yp = bayesian_optimisation(n_iters, sample_loss, bounds, n_pre_samples=n_pre_samples)

            opt_results += [get_optimal_loss(lang, args.is_devoicing, artificial, token_map, concept_ids,
                                             ipa_to_concepts, xp, yp, args)]

            write_csv(results, '%s/artificial__%s__baysian-results.csv' % (args.rfolder, args.model))
            write_csv(opt_results, '%s/artificial__%s__opt-results.csv' % (args.rfolder, args.model))

    write_csv(results, '%s/artificial__%s__baysian-results-final.csv' % (args.rfolder, args.model))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='artificial/%s/bayes-opt')
    assert args.data == 'northeuralex', 'this script should only be run with northeuralex data'
    fill_artificial_args(args)
    optimize_languages(args)
