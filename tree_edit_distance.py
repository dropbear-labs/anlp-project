from zss import simple_distance, Node

from common import Sentence


class TreeEditDistance:
    def distance(self, s1: Sentence, s2: Sentence) -> float:
        return [simple_distance(s1.tree(), s2.tree())]
