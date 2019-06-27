import torch
import torch.nn as nn

from model.lstm import IpaLM
from model.phoible import PhoibleEmbeddings


class PhoibleLookupLM(IpaLM):
    def __init__(self, vocab_size, hidden_size, token_map, **kwargs):
        super().__init__(vocab_size, hidden_size, **kwargs)

        self.embs = PhoibleEmbeddings(token_map, self.embedding_size)
        self.lstm = nn.LSTM(
            2 * self.embedding_size, hidden_size, self.nlayers,
            dropout=(self.dropout_p if self.nlayers > 1 else 0), batch_first=True)

    def forward(self, x, h_old=None):
        x_emb_multi = self.embs(x)
        x_emb_single = self.embedding(x)

        x_emb = self.dropout(torch.cat([x_emb_single, x_emb_multi], dim=-1))

        c_t, h_t = self.lstm(x_emb, h_old)
        c_t = self.dropout(c_t).contiguous()

        logits = self.out(c_t)
        return logits, h_t
