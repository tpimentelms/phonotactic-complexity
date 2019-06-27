from model.phoible import PhoibleEmbeddings
from model.shared_lstm import SharedIpaLM


class SharedPhoibleLM(SharedIpaLM):
    def __init__(self, languages, vocab_size, hidden_size, token_map, nlayers=1, dropout=0.1, embedding_size=None):
        embedding_size = embedding_size if embedding_size is not None else hidden_size
        super().__init__(
            languages, vocab_size, hidden_size, nlayers=nlayers, dropout=dropout, embedding_size=embedding_size)

        self.embs = PhoibleEmbeddings(token_map, self.embedding_size)

    def forward(self, x, lang, h_old=None):
        x_emb = self.embs(x)

        lm = getattr(self, 'lm_%s' % lang)
        return lm(x_emb, h_old=h_old)
