import nltk
import string
from dataclasses import dataclass
from zss import simple_distance, Node
from typing import List

punct = set(string.punctuation)


@dataclass
class Sentence:
    s: str
    ts: str

    def lowercase_tokens(self):
        return [x for x in nltk.wordpunct_tokenize(self.s.lower()) if len(set(x) & punct) == 0]

    def tree(self):
        t = nltk.tree.Tree.fromstring(self.ts)
        def _to_tree(n):
            if isinstance(n, str):
                return Node(n)
            children = [_to_tree(ch) for ch in n]
            node = Node(n.label())
            for ch in children:
                node.addkid(ch)
            return node
        return _to_tree(t)


class Multi:
    def __init__(self, models):
        self.models = models

    def distance(self, s1: Sentence, s2: Sentence) -> List[float]:
        rval = []
        for model in self.models:
            rval.extend(model.distance(s1, s2))
        return rval
