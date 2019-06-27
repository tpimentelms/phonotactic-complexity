import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
sys.path.append('./')
from util import constants

aspect = {
    'size': 6.5,
    'font_scale': 2.5,
    'labels': False,
    'name_suffix': 'small__shared',
    'ratio': 1.625,
}

sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
mpl.rc('figure', figsize=(aspect['size'] * aspect['ratio'], aspect['size'] * 1.2))
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

df = pd.read_csv('%s/compiled-results_avg.tsv' % constants.rfolder, delimiter='\t')
x = df['lstm'] * df['avg_len']

sns.distplot(x, bins=10, label='Histogram density (10 bins)', norm_hist=True, kde=False)
sns.distplot(x, bins=100, label='Histogram density (100 bins)', norm_hist=True, kde=False)
sns.distplot(x, bins=100, label='Kernel Density Estimate', hist=False)

plt.ylabel('Density')
plt.xlabel('Complexity (Bits per word)')

plt.legend(labelspacing=0.2)
plt.xlim([12, 26])
plt.ylim([0, 1.25])
plt.tight_layout()
plt.savefig('plot/kde_word_complexity.pdf', bbox_inches='tight')
plt.show()
