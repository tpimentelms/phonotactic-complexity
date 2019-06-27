import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy

from model.lstm import IpaLM


class PhoibleEmbeddings(nn.Module):
    def __init__(self, token_map, embedding_size, data_path='datasets'):
        super().__init__()
        self.df = pd.read_csv('%s/phoible-segments-features.tsv' % data_path, delimiter='\t', encoding='utf-8')
        self.create_multi_hot(token_map)
        self.weight = nn.Parameter(torch.Tensor(self.hot_size, embedding_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def create_multi_hot(self, token_map):
        unrecognized = self.get_unrecognized(token_map.keys())
        positions, size = self.get_positions()

        self.hot_size = size + 1
        multi_hot = np.zeros((len(token_map), self.hot_size))
        for token, idx in token_map.items():
            emb = self.create_multi_hot_single(token, size, positions, unrecognized)
            multi_hot[idx, :] = emb

        with torch.no_grad():
            hot_tensor = torch.Tensor(multi_hot)
            self.multi_hot = nn.Parameter(hot_tensor)

    def get_unrecognized(self, tokens):
        unrecognized = []
        for token in tokens:
            row = self.df[self.df.segment == token]

            if len(row) > 1:
                unrecognized += [token]
                print('Phoible Error: more than one row for IPA %s' % token)
                import ipdb; ipdb.set_trace()
            elif len(row) == 0:
                unrecognized += [token]
                continue

        return unrecognized

    def get_positions(self):
        temp_df = copy.copy(self.df)
        positions, size = {}, 0
        del temp_df['segment']
        for col in temp_df.columns:
            unique_vals = temp_df[col].unique()
            positions[col] = {val: size + i for i, val in enumerate(unique_vals)}
            size += len(unique_vals)

        return positions, size

    def create_multi_hot_single(self, token, size, positions, unrecognized):
        emb = np.zeros(size + 1)
        if token in unrecognized:
            emb[-1] = 1
            return emb

        row = self.df[self.df.segment == token]
        del row['segment']

        for col in row.columns:
            val = row[col].iloc[0]
            pos = positions[col][val]
            emb[pos] = 1

        return emb

    def get_multi_hot(self, input):
        phoible_input = F.embedding(input, self.multi_hot)
        return phoible_input

    def forward(self, x):
        phoible_input = self.get_multi_hot(x)
        x_emb = (phoible_input @ self.weight) / phoible_input.sum(2, keepdim=True)
        return x_emb


class PhoibleLM(IpaLM):
    def __init__(self, vocab_size, hidden_size, token_map, **kwargs):
        super().__init__(vocab_size, hidden_size, **kwargs)

        self.embs = PhoibleEmbeddings(token_map, self.embedding_size)

    def forward(self, x, h_old=None):
        x_emb = self.embs(x)

        c_t, h_t = self.lstm(x_emb, h_old)
        c_t = self.dropout(c_t).contiguous()

        logits = self.out(c_t)
        return logits, h_t
