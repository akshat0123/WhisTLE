from collections import defaultdict

import torch


class TrigramLM:

    def __init__(self, vocab_size, smooth=False):
        self.unigram = defaultdict(int)
        self.bigram = defaultdict(lambda: defaultdict(int))
        self.trigram = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.vocab_size = vocab_size
        self.smooth = smooth 
        self.trained = False

    def add(self, tokens):
        if not self.trained:
            for i in range(len(tokens)):
                self.unigram[tokens[i]] += 1

                if i > 0:
                    self.bigram[tokens[i-1]][tokens[i]] += 1

                if i > 1:
                    self.trigram[tokens[i-2]][tokens[i-1]][tokens[i]] += 1
        else:
            raise RuntimeError("Tokens cannot be added to trained N-gram model."
                               "Please initialize a new model.")

    def _train_unigram(self):
        # Calculate probabilities
        total = 0
        for one in self.unigram:
            total += self.unigram[one]

        for one in self.unigram:
            self.unigram[one] /= total

        # Cache prob vector
        if not self.smooth:
            self.unigram_probs = torch.zeros((1, 1, self.vocab_size))
        else:
            self.unigram_probs = torch.full((1, 1, self.vocab_size), 1.0/total)

        for one in self.unigram:
            self.unigram_probs[0, 0, one] += self.unigram[one]

    def _train_bigram(self):
        # Calculate probabilities
        for one in self.bigram:

            total = 0
            for two in self.bigram[one]:
                total += self.bigram[one][two]

            for two in self.bigram[one]:
                self.bigram[one][two] /= total

        # Cache prob vectors
        self.bigram_probs = {} 
        for one in self.bigram:

            total = len(self.bigram[one])
            if not self.smooth:
                self.bigram_probs[one] = torch.zeros((1, 1, self.vocab_size))
            else:
                self.bigram_probs[one] = torch.full((1, 1, self.vocab_size), 1.0/total)

            for two in self.bigram[one]:
                self.bigram_probs[one][0, 0, two] += self.bigram[one][two]

    def _train_trigram(self):
        # Calculate probabilities
        for one in self.trigram:
            for two in self.trigram[one]:

                total = 0
                for three in self.trigram[one][two]:
                    total += self.trigram[one][two][three]

                for three in self.trigram[one][two]:
                    self.trigram[one][two][three] /= total

        # Cache log-prob vectors
        self.trigram_probs = {}
        for one in self.trigram:
            for two in self.trigram[one]:

                total = len(self.trigram[one][two])

                if one in self.trigram_probs:

                    if not self.smooth:
                        self.trigram_probs[one][two] = torch.zeros((1, 1, self.vocab_size))
                    else:
                        self.trigram_probs[one][two] = torch.full((1, 1, self.vocab_size), 1.0/total)

                else:

                    if not self.smooth:
                        vec = torch.zeros((1, 1, self.vocab_size))
                    else:
                        vec = torch.full((1, 1, self.vocab_size), 1.0/total)

                    self.trigram_probs[one] = { two: vec }

                for three in self.trigram[one][two]:
                    self.trigram_probs[one][two][0, 0, three] += self.trigram[one][two][three]

    def train(self):
        self._train_unigram()
        self._train_bigram()
        self._train_trigram()
        self.trained = True

    def __call__(self, x=[]):
        if len(x) >= 2 and x[-2] in self.trigram and \
           x[-1] in self.trigram[x[-2]]:
            log_probs = self.trigram_probs[x[-2]][x[-1]]

        elif len(x) >= 1 and x[-1] in self.bigram:
            log_probs = self.bigram_probs[x[-1]]

        else:
            log_probs = self.unigram_probs

        return log_probs 
