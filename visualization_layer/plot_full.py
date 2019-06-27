import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.append('./')
from util import constants


aspect = {
    'size': 7,
    'font_scale': 1.6,
    'labels': True,
    'ratio': 2.125,
}
vals = ['lstm', 'ngram']

sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})


def label_points(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + .02, point['y'], str(point['val']), color='C5')


df = pd.read_csv('%s/compiled-results_avg.tsv' % (constants.rfolder), delimiter='\t')
names_map = {
    'lstm': '$LSTM$',
    'ngram': 'trigram',
}

frames = []
for val in vals:
    df_new = df[['lang', 'avg_len']].copy()
    df_new['test_loss'] = df[val]
    df_new['Model'] = names_map[val]
    df_new.reset_index(level=0, inplace=True)

    frames += [df_new]
data = pd.concat(frames)

use_legend = len(vals) > 1
fig = sns.lmplot(
    'avg_len', 'test_loss', data, hue='Model', palette='muted',
    height=aspect['size'], aspect=aspect['ratio'], legend_out=False, truncate=False, legend=use_legend)

if use_legend:
    plt.legend(title='Model', loc='upper right')

if aspect['labels']:
    label_points(data.avg_len, data.test_loss, data.lang, plt.gca())


plt.xlim([4.5, 8.8])
plt.xlabel('Average Length (# IPA tokens)')
plt.ylabel('Cross Entropy (bits per phoneme)')
fig.savefig('plot/full_complexity.pdf')
plt.show()
