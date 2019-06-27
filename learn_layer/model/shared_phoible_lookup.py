import torch
import torch.nn as nn

from model.phoible import PhoibleEmbeddings
from model.shared_lstm import SharedIpaLM


class SharedPhoibleLookupLM(SharedIpaLM):
    def __init__(self, languages, vocab_size, hidden_size, token_map, nlayers=1, dropout=0.1, embedding_size=None):
        embedding_size = embedding_size if embedding_size is not None else hidden_size
        super().__init__(
            languages, vocab_size, hidden_size, nlayers=nlayers, dropout=dropout, embedding_size=2 * embedding_size)
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.embs = PhoibleEmbeddings(token_map, self.embedding_size)

    def forward(self, x, lang, h_old=None):
        x_emb_multi = self.embs(x)
        x_emb_single = self.embedding(x)

        x_emb = self.dropout(torch.cat([x_emb_single, x_emb_multi], dim=-1))

        lm = getattr(self, 'lm_%s' % lang)
        return lm(x_emb, h_old=h_old)
