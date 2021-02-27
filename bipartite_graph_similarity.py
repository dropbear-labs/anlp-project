from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import sklearn.preprocessing as pre

import networkx as nx

from common import Sentence, sigmoid


@dataclass
class BipartiteGraphSim:
    vectors: Any
    word_to_idx: dict

    def distance(self, s1: Sentence, s2: Sentence) -> float:
        B = nx.Graph()

        s1_tokens = s1.lowercase_tokens()
        s2_tokens = s2.lowercase_tokens()

        top_nodes = [(0, idx) for idx in range(len(s1_tokens))]
        bottom_nodes = [(1, idx) for idx in range(len(s2_tokens))]
        B.add_nodes_from(top_nodes, bipartite=0)
        B.add_nodes_from(bottom_nodes, bipartite=1)

        for idx1, t1 in enumerate(s1_tokens):
            for idx2, t2 in enumerate(s2_tokens):
                # use negative of similarity because we will do minimum weight matching
                B.add_edge((0, idx1), (1, idx2), weight=-self.sim(t1, t2))

        matching = nx.bipartite.matching.minimum_weight_full_matching(B)
        edges = [(v_from, v_to) for v_from, v_to in matching.items() if v_from[0] == 0]
        sum_sim = sum(-B[v_from][v_to]["weight"] for v_from, v_to in edges)
        return [sum_sim / len(s2_tokens)]

    @classmethod
    def load_random(cls):
        return cls.load()

    def params(self):
        return {}

    @classmethod
    def load(cls, **extra_constructor_args):
        words, vs = [], []
        f = "/home/nrg/datasets/glove.6B.100d.txt"
        for idx, line in enumerate(open(f)):
            if idx >= 60000:
                break
            word, *_vals = line.strip().split()
            words.append(word)
            vals = [float(v) for v in _vals]
            vs.append(np.array(vals))
        word_to_index = {w: i for i, w in enumerate(words)}
        vectors = np.vstack(vs)
        #vectors = pre.normalize(np.vstack(vs))
        return cls(vectors, word_to_index, **extra_constructor_args)

    def sim(self, w1, w2):
        if w1 not in self.word_to_idx or w2 not in self.word_to_idx:
            return 0
        return np.dot(
            self.vectors[self.word_to_idx[w1]],
            self.vectors[self.word_to_idx[w2]]
        )


@dataclass
class TunedBipartiteGraphSim(BipartiteGraphSim):
    w: float = 1
    b: float = 0

    @classmethod
    def load_random(cls):
        return cls.load(
            w=random.random()*2 - 1.5,
            b=random.random()*2 - 1,
        )

    def params(self):
        return {"w": self.w, "b": self.b}

    def sim(self, w1, w2):
        if w1 not in self.word_to_idx or w2 not in self.word_to_idx:
            return 0
        dot = np.dot(
            self.vectors[self.word_to_idx[w1]],
            self.vectors[self.word_to_idx[w2]]
        )
        return sigmoid(self.w*dot + self.b)


# @dataclass
# class TunedBipartiteRBFGraphSim(BipartiteGraphSim):
#     eps: float = 1

#     @classmethod
#     def load_random(cls):
#         return cls.load(
#             w=random.random()*20 - 10,
#             b=random.random()*20 - 10,
#         )

#     def params(self):
#         return {"w": self.w, "b": self.b}

#     def sim(self, w1, w2):
#         if w1 not in self.word_to_idx or w2 not in self.word_to_idx:
#             return 0
#         dot = np.dot(
#             self.vectors[self.word_to_idx[w1]],
#             self.vectors[self.word_to_idx[w2]]
#         )
#         return np.exp()
