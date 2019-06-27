import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.append('./')
from util import constants

aspect = {
    'size': 6.5,
    'font_scale': 2.5,
    'labels': False,
    'ratio': 1.625,
}
models = ['ngram', 'lstm']

sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
mpl.rc('figure', figsize=(aspect['size'] * aspect['ratio'], aspect['size'] * aspect['ratio']))
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})


def get_artificial_df(folder):
    ngram_file = '%s/artificial__ngram__full-results.csv' % folder
    lstm_file = '%s/artificial__lstm__full-results.csv' % folder

    df_artificial_ngram = pd.read_csv(ngram_file)
    df_artificial_lstm = pd.read_csv(lstm_file)

    df_artificial_ngram['Model'] = 'trigram'
    df_artificial_lstm['Model'] = 'LSTM'

    df_models = []
    df_models += [df_artificial_ngram] if 'ngram' in models else []
    df_models += [df_artificial_lstm] if 'lstm' in models else []

    df = pd.concat(df_models, sort=False)
    df = df.sort_values(['Model', 'artificial'], ascending=[False, True])

    df_art = df[df.artificial].copy()
    df_art['artificial'] = df_art['test_loss']
    df_art = df_art[['Model', 'lang', 'artificial', 'fold']]
    df_nat = df[~df.artificial].copy()
    df_nat['natural'] = df_nat['test_loss']
    df_nat = df_nat[['Model', 'lang', 'natural', 'fold']]

    df_final = df_art.set_index(['Model', 'lang', 'fold']) \
        .join(df_nat.set_index(['Model', 'lang', 'fold'])) \
        .reset_index()
    return df_final


def get_artificial_df_avg(folder):
    df_final = get_artificial_df(folder)
    df_final = df_final.groupby(['Model', 'lang']).agg('mean').reset_index()
    return df_final


df_final = get_artificial_df_avg(constants.rfolder_artificial_harmony)

sns.scatterplot(x='natural', y='artificial', data=df_final, hue='Model', style='Model', s=500, markers=['P', 'o'])

delta_y = df_final.artificial.max() - df_final.artificial.min()
y = [df_final.artificial.min() + x * delta_y for x in range(0, 2)]

min_y = df_final.artificial.min()
min_x = df_final.natural.min()
max_y = df_final.artificial.max()
max_x = df_final.natural.max()
min_range = min(min_y, min_x) - .1
max_range = max(max_y, max_x) + .1
plot_range = [min_range, max_range]

plt.plot(plot_range, plot_range, 'C2--', linewidth=2, alpha=0.8)

plt.legend(labelspacing=0.2, markerscale=4)

plt.xlim(plot_range)
plt.xticks(np.arange(2.75, 4.8, step=0.5))
plt.ylim(plot_range)
plt.yticks(np.arange(2.75, 4.8, step=0.5))
plt.xlabel('Original Language')
plt.ylabel('Artificial Language')

plt.tight_layout()
plt.savefig('plot/scatterplot_harmony.pdf', bbox_inches='tight')
plt.show()
