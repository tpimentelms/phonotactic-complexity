import argparse
import re
import glob
import pandas as pd

parser = argparse.ArgumentParser(description='Phoneme LM')

# Data
parser.add_argument('--data', type=str, default='northeuralex',
                    choices=['northeuralex', 'celex'],
                    help='Dataset used. (default: northeuralex)')
parser.add_argument('--run-mode', type=str, default='cv',
                    choices=['normal', 'bayes-opt', 'cv', 'opt'],
                    help='Path where data is stored.')

args = parser.parse_args()


def preparse_df(df):
    df = df[~(df.lang == 'cmn')]
    return df


def read_df(fname):
    df = pd.read_csv(fname, delimiter=',')
    return preparse_df(df)


def get_file_name(is_bayesian, is_cv):
    file_name = 'opt-results' if is_bayesian else 'results-final'
    file_name = 'full-results' if is_cv else file_name
    return file_name


def get_file_names(folder, run_mode):
    is_bayesian = (run_mode == 'bayes-opt')
    is_cv = (run_mode == 'cv')

    file_name = get_file_name(is_bayesian, is_cv)

    files_pattern_simpl = '%s/*__%s.csv' % (folder, file_name)
    files_simpl = glob.glob(files_pattern_simpl)
    index = ['lang', 'fold'] if is_cv else ['lang']

    file_names = ['bayesian-shared-results', 'opt-results'] if is_bayesian else ['*results-final']
    file_names = ['full-results'] if is_cv else file_names
    file_pattern = '(bayesian-shared-results|opt-results)' if is_bayesian else '(shared-)?results-final'
    file_pattern = 'full-results' if is_cv else file_pattern

    return file_names, file_pattern, files_simpl, index


data = args.data
run_mode = args.run_mode

folder = 'results/%s/%s/orig' % (data, run_mode)
new_folder = 'results/%s/%s' % (data, run_mode)

file_names, file_pattern, files_simpl, index = get_file_names(folder, run_mode)


files_glob_pattern = ['%s/*__%s.csv' % (folder, x) for x in file_names]
files_re = '%s/(.*)__(%s).csv' % (folder, file_pattern)
files = [y for x in files_glob_pattern for y in glob.glob(x)]
base_file = [x for x in files_simpl if 'unigram' in x][0]
print(files)

df = read_df(base_file)
df['avg_len'] = df['full_avg_len']
df_comp = df[index + ['avg_len']].copy()
df_comp.set_index(index, inplace=True)

df_comp = df_comp.join(df[index].set_index(index), on=index)


for f in files:
    df = read_df(f)
    m = re.search(files_re, f)
    model = m.group(1)

    df = df.drop(df[df['lang'] == 'full'].index)

    df.set_index(index, inplace=True)
    if 'avg_len' not in df.columns:
        df = df.join(df_comp[['avg_len']], on=index)

    df.to_csv('%s/%s__complexity.tsv' % (new_folder, model), sep='\t')
    df_res = df.groupby('lang').agg('mean')
    df_res.to_csv('%s/%s__complexity_avg.tsv' % (new_folder, model), sep='\t')

    df[model] = df['test_loss']
    df_comp = df_comp.join(df[[model]], on=index)

df_comp.to_csv('%s/compiled-results.tsv' % (new_folder), sep='\t')

df_res = df_comp.groupby('lang').agg('mean')
df_res.to_csv('%s/compiled-results_avg.tsv' % (new_folder), sep='\t')
