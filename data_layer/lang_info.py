import pandas as pd


class LangInfo(object):
    def __init__(self):
        self.df = pd.read_csv('datasets/languages.csv', encoding='utf-8')

    def get_family(self, lang):
        row = self.df[self.df.iso_code == lang]
        assert len(row) <= 1, 'Language Error: more than one row for language %s' % lang
        assert len(row) != 0, 'Language Error: language not found %s' % lang

        return row.family.iloc[0]
