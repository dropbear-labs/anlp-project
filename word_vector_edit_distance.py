from dataclasses import dataclass
from typing import Any


"""
Edit Distance Based 71.9

Use test and development (10k each)
"""
import random
import numpy as np
import sklearn.preprocessing as pre
import distance
import nltk
import string

from common import Sentence


def levenshtein(s1, s2, del_cost, insert_cost, subs_cost, normalize=False):
    table = {}

    for i in range(1, len(s1)+1):
        table[(i, 0)] = i * del_cost

    for j in range(1, len(s2)+1):
        table[(0, j)] = i * insert_cost

    for j in range(1, len(s2)+1):
        for i in range(1, len(s1)+1):
            if s1[i-1] == s2[j-1]:
                this_subs_cost = 0
            else:
                this_subs_cost = subs_cost

            table[(i, j)] = min(
                table.get((i-1, j), 0) + del_cost,
                table.get((i, j-1), 0) + insert_cost,
                table.get((i-1, j-1), 0) + this_subs_cost
            )

    if normalize:
        return table[(len(s1), len(s2))] / max(len(s1), len(s2))
    return table[(len(s1), len(s2))]


class EditDistanceI:
    def distance(self, s1: Sentence, s2: Sentence) -> float:
        raise NotImplementedError


class NormalizedLevenshtein(EditDistanceI):
    def distance(self, s1: Sentence, s2: Sentence) -> float:
        return [distance.nlevenshtein(s1.lowercase_tokens(), s2.lowercase_tokens())]


@dataclass
class TunedNormalizedLevenshtein(EditDistanceI):
    del_cost: float = 1
    insert_cost: float = 1
    subs_cost: float = 1
    normalize: bool = False

    def distance(self, s1: Sentence, s2: Sentence) -> float:
        return [
            levenshtein(
                s1.lowercase_tokens(),
                s2.lowercase_tokens(),
                self.del_cost,
                self.insert_cost,
                self.subs_cost,
                self.normalize,
            )
        ]


def sigmoid(v):
    e_v = np.exp(v)
    return e_v / (e_v + 1)


@dataclass
class WVEditDistance:
    w: float # \in [0, 20]
    b: float # \in [-1, 1]
    lambda_: float  # \in [0,1]
    mu: float       # \in [0,1]
    vectors: Any
    word_to_idx: dict

    def __repr__(self):
        return f"WVEditDistance(w={self.w}, b={self.b}, lambda_={self.lambda_}, mu={self.mu})"

    @classmethod
    def load(cls, w: float, b: float, lambda_: float, mu: float):
        words, vs = [], []
        f = "/home/nrg/potsdam/anlp/a1/glove.6B/glove.6B.50d.txt"
        for idx, line in enumerate(open(f)):
            if idx >= 60000:
                break
            word, *_vals = line.strip().split()
            words.append(word)
            vals = [float(v) for v in _vals]
            vs.append(np.array(vals))
        word_to_index = {w: i for i, w in enumerate(words)}
        vectors = pre.normalize(np.vstack(vs))
        return cls(w, b, lambda_, mu, vectors, word_to_index)

    @classmethod
    def load_default(cls):
        return cls.load(1, 1, 1, 1)

    @classmethod
    def load_random(cls):
        mu = random.random() * 3 - 2
        return cls.load(
            random.randrange(0, 20),
            random.random() * 2 - 1,
            random.random() / (1-mu),
            mu
        )

    def sim(self, w1, w2):
        if w1 == w2:
            return 1
        if w1 not in self.word_to_idx or w2 not in self.word_to_idx:
            return 0
        dot = np.dot(
            self.vectors[self.word_to_idx[w1]],
            self.vectors[self.word_to_idx[w2]]
        )
        return sigmoid(self.w * dot + self.b)

    def wv_levenshtein(self, s1, s2):
        table = {}

        all_sims = []
        for w1 in s1:
            these_sims = []
            for w2 in s2:
                these_sims.append(self.sim(w1, w2))
            all_sims.append(these_sims)

        for i in range(1, len(s1)+1):
            table[(i, 0)] = table.get((i-1, 0), 0) + 1 - self.lambda_ * max(all_sims[i-1][idx] for idx in range(len(s2)) if idx != i-1) + self.mu

        for j in range(1, len(s2)+1):
            table[(0, j)] = table.get((0, j-1), 0) + 1 - self.lambda_ * max(all_sims[idx][j-1] for idx in range(len(s1)) if idx != j-1) + self.mu

        for j in range(1, len(s2)+1):
            for i in range(1, len(s1)+1):
                insert_cost = 1 - self.lambda_ * max(all_sims[idx][j-1] for idx in range(len(s1)) if idx != i-1) + self.mu
                del_cost = 1 - self.lambda_ * max(all_sims[i-1][idx] for idx in range(len(s2)) if idx != j-1) + self.mu
                this_subs_cost = 2 - 2 * all_sims[i-1][j-1]

                table[(i, j)] = min(
                    (table.get((i-1, j), 0) + del_cost),
                    (table.get((i, j-1), 0) + insert_cost),
                    (table.get((i-1, j-1), 0) + this_subs_cost)
                )

        return table[(len(s1), len(s2))]

    def distance(self, s1: Sentence, s2: Sentence) -> float:
        return [
            self.wv_levenshtein(
                s1.lowercase_tokens(),
                s2.lowercase_tokens(),
            )
        ]
