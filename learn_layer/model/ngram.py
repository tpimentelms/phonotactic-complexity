import numpy as np
from nltk.model import build_vocabulary, count_ngrams, BaseNgramModel, MLENgramModel


class LaplaceUnigramModel(BaseNgramModel):
    """Provides Lidstone-smoothed scores.
    In addition to initialization arguments from BaseNgramModel also requires
    a number by which to increase the counts, gamma.
    """

    def __init__(self, vocab_size, *args):
        super().__init__(*args)
        self.gamma = 1.
        self.gamma_norm = vocab_size * 1.0

        self.unigrams = self.ngram_counter.unigrams

    def get_entropy(self, text):
        entropies = [self.entropy(phrase) for phrase in text]
        return np.mean(entropies)

    def score(self, word, context=None):
        word_count = self.unigrams[word]
        ctx_count = self.unigrams.N()
        return (word_count + self.gamma) / (ctx_count + self.gamma_norm)

    def entropy(self, text, average=True):
        normed_text = (self._check_against_vocab(word) for word in text)
        H = 0.0  # entropy is conventionally denoted by "H"
        processed_ngrams = 0
        for i in range(0, len(text)):
            word = text[i]
            H += self.logscore(word, None)
            processed_ngrams += 1

        if average:
            return - (H / processed_ngrams)
        else:
            return - H, processed_ngrams


class NGramModel(BaseNgramModel):
    def __init__(self, text, vocab_size, order=3, params=None):
        self.order = order
        self._counters, self._models = [], {}

        self.params = np.array(params if params is not None else [1. / order] * order)
        assert len(self.params) == order, 'NGramModel params needs to have same dimension as order'

        vocab = build_vocabulary(1, *text)
        counter = count_ngrams(1, vocab, text, pad_left=False, pad_right=False)
        self._models[0] = LaplaceUnigramModel(vocab_size, counter)

        for ctx_size in range(1, order):
            counter = count_ngrams(ctx_size + 1, vocab, text, pad_left=False, pad_right=False)
            self._counters += [counter]
            self._models[ctx_size] = MLENgramModel(counter)

        super().__init__(self._counters[-1])

    def get_entropy(self, text):
        entropies = [self.entropy(phrase) for phrase in text]
        return np.mean(entropies)

    def score(self, word, context):
        context = self.check_context(context)
        _n = len(context) + 1

        scores = [self._models[ctx_size].score(word, context[-(ctx_size):]) for ctx_size in range(_n)]
        scores = np.array(scores)

        norm_param = self.params[:_n] / np.sum(self.params[:_n], keepdims=True)
        return norm_param @ scores

    def entropy(self, text, average=True):
        normed_text = (self._check_against_vocab(word) for word in text)
        H = 0.0     # entropy is conventionally denoted by "H"
        processed_ngrams = 0
        for i in range(1, min(len(text), self.order)):
            context, word = tuple(text[:i]), text[i]
            H += self.logscore(word, context)
            processed_ngrams += 1

        for ngram in self.ngram_counter.to_ngrams(normed_text):
            context, word = tuple(ngram[:-1]), ngram[-1]
            H += self.logscore(word, context)
            processed_ngrams += 1

        if average:
            return - (H / processed_ngrams)
        else:
            return - H, processed_ngrams
