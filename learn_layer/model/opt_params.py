import pandas as pd


def _get_opt_params(fname, lang, delimiter='\t'):
    results = pd.read_csv(fname, delimiter=delimiter)
    instance = results[results['lang'] == lang]

    embedding_size = int(instance['embedding_size'].item())
    hidden_size = int(instance['hidden_size'].item())
    nlayers = int(instance['nlayers'].item())
    dropout = instance['dropout'].item()

    return embedding_size, hidden_size, nlayers, dropout


def get_opt_params(model, lang, data='northeuralex', reverse=False):
    folder_suffix = '' if not reverse else '_inv'
    fname = 'results/%s%s/bayes-opt/orig/%s__opt-results.csv' % (data, folder_suffix, model)
    return _get_opt_params(fname, lang, delimiter=',')


def get_shared_opt_params(model, data='northeuralex', reverse=False):
    folder_suffix = '' if not reverse else '_inv'
    fname = 'results/%s%s/bayes-opt/orig/%s__bayesian-shared-results.csv' % (data, folder_suffix, model)
    return _get_opt_params(fname, 'full', delimiter=',')


def _get_artificial_opt_params(fname, lang, is_artificial, delimiter='\t'):
    results = pd.read_csv(fname, delimiter=delimiter)
    instance = results[(results['lang'] == lang) & (results['artificial'] == is_artificial)]

    embedding_size = int(instance['embedding_size'].item())
    hidden_size = int(instance['hidden_size'].item())
    nlayers = int(instance['nlayers'].item())
    dropout = instance['dropout'].item()

    return embedding_size, hidden_size, nlayers, dropout


def get_artificial_opt_params(model, lang, is_artificial, artificial_type, data='northeuralex'):
    fname = 'results/%s/artificial/%s/bayes-opt/orig/artificial__%s__opt-results.csv' % (data, artificial_type, model)
    return _get_artificial_opt_params(fname, lang, is_artificial, delimiter=',')
