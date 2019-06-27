import copy

import torch.nn as nn


class IpaLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, nlayers=1, dropout=0.1, embedding_size=None):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size
        self.dropout_p = dropout
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(
            self.embedding_size, hidden_size, nlayers, dropout=(dropout if nlayers > 1 else 0), batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, vocab_size)

        self._cuda = False
        self.best_state_dict = None

    def forward(self, x, h_old=None):
        x_emb = self.dropout(self.embedding(x))
        c_t, h_t = self.lstm(x_emb, h_old)
        c_t = self.dropout(c_t).contiguous()

        logits = self.out(c_t)
        return logits, h_t

    def initHidden(self, bsz=1):
        weight = next(self.parameters()).data
        return weight.new(self.nlayers, bsz, self.hidden_size).zero_(), \
            weight.new(self.nlayers, bsz, self.hidden_size).zero_()

    def cuda(self):
        super().cuda()
        self._cuda = True

    def cpu(self):
        super().cpu()
        self._cuda = False

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)
