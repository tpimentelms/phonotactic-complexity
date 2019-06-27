import pandas as pd
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
}
size = 'small'

sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})


df = pd.read_csv('%s/lang_inventory.csv' % constants.rfolder_inventory)
vals = ['consonant', 'vowel', 'inventory']

frames = []
for val in vals:
    df_new = df[['lang', 'avg_len']].copy()
    df_new['count_unique'] = df[val]
    df_new['# Unique'] = val
    df_new.reset_index(level=0, inplace=True)

    frames += [df_new]
data = pd.concat(frames)


fig = sns.lmplot(
    'avg_len', 'count_unique', data, hue='# Unique', palette='muted',
    height=aspect['size'] * 1.2, aspect=1.625 / 1.2, legend_out=False)

plt.legend(title='# Unique', labelspacing=0.2)
plt.xlim([4.35, 8.8])
plt.ylim([0, 79])
plt.xlabel('Average Length (# IPA tokens)')
plt.ylabel('Inventory Size (# Phonemes)')
fig.savefig('plot/inventory_complexity.pdf')
plt.show()
