import pandas as pd


class PhoibleInfo(object):
    def __init__(self):
        self.df = pd.read_csv('datasets/phoible-segments-features.tsv', delimiter='\t', encoding='utf-8')

    def get_types(self, input):
        results, unrecognized = [], []

        for token in input:
            row = self.df[self.df.segment == token]
            if len(row) > 1:
                unrecognized += [token]
                print('Phoible Error: more than one row for IPA %s' % token)
                continue
            elif len(row) == 0:
                unrecognized += [token]
                continue

            if row.consonantal.iloc[0] == '+':
                results += ['consonant']
            elif row.consonantal.iloc[0] == '-':
                results += ['vowel']
            elif row.consonantal.iloc[0] == '0' and row.tone.iloc[0] == '+':
                results += ['tone']
            elif row.consonantal.iloc[0] == '0' and row.standaloneSymbol.iloc[0] == '+':
                results += ['symbol']
            else:
                print('Phoible Error: Unrecognized type for IPA %s' % token)

        return results, unrecognized

    def count_types(self, input):
        types, unrecognized = self.get_types(input)
        consonant = sum([x == 'consonant' for x in types])
        vowel = sum([x == 'vowel' for x in types])
        tone = sum([x == 'tone' for x in types])
        symbol = sum([x == 'symbol' for x in types])
        unrecognized = len(unrecognized)

        return consonant, vowel, tone, symbol, unrecognized
