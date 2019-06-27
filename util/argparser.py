import argparse
from . import util

parser = argparse.ArgumentParser(description='Phoneme LM')

# Data
parser.add_argument('--data', type=str, default='northeuralex',
                    help='Dataset used. (default: northeuralex)')
parser.add_argument('--data-path', type=str, default='datasets',
                    help='Path where data is stored.')
parser.add_argument('--artificial-type', default='harmony',
                    choices=['harmony', 'devoicing'],
                    help='Artificial data type used. (default: harmony)')

# Model
parser.add_argument('--model', default='lstm',
                    choices=['lstm', 'phoible', 'phoible-lookup', 'unigram', 'ngram',
                             'shared-lstm', 'shared-phoible', 'shared-phoible-lookup'],
                    help='Model used. (default: lstm)')
parser.add_argument('--opt', action='store_true', default=False,
                    help='Should use optimum parameters in training.')
parser.add_argument('--cv', action='store_true', default=False,
                    help='Should use cross validation.')


# Others
parser.add_argument('--results-path', type=str, default='results',
                    help='Path where results should be stored.')
parser.add_argument('--seed', type=int, default=7,
                    help='Seed for random algorithms repeatability (default: 7)')


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)


def get_default(*args, **kwargs):
    return parser.get_default(*args, **kwargs)


def parse_args(*args, csv_folder='', **kwargs):
    args = parser.parse_args(*args, **kwargs)
    if 'artificial' in csv_folder:
        csv_folder = csv_folder % (args.artificial_type)
    args.ffolder = '%s/%s' % (args.data_path, args.data)  # Data folder
    args.rfolder_base = '%s/%s' % (args.results_path, args.data)  # Results base folder
    args.rfolder = '%s/%s/orig' % (args.rfolder_base, csv_folder)  # Results folder
    util.mkdir(args.rfolder)
    util.config(args.seed)
    return args
