from dataclasses import dataclass
from common import Sentence


from distance_metrics import lcs
import distance

@dataclass
class LongestCommonSubsequence:
    def distance(self, s1: Sentence, s2: Sentence) -> float:
        return [
            lcs.llcs(
                s1.lowercase_tokens(),
                s2.lowercase_tokens(),
            )
        ]

    @classmethod
    def load_random(cls):
        return cls.load()

    @classmethod
    def load(cls):
        return cls()

    def params(self):
        return {}


@dataclass
class Bakkelund:
    def distance(self, s1: Sentence, s2: Sentence) -> float:
        return [
            lcs.bakkelund(
                s1.lowercase_tokens(),
                s2.lowercase_tokens(),
            )
        ]

    @classmethod
    def load_random(cls):
        return cls.load()

    @classmethod
    def load(cls):
        return cls()

    def params(self):
        return {}
