import json
from pathlib import Path
from collections import Counter

import nltk
import sklearn.model_selection as sel
import sklearn.linear_model as lm
import sklearn.metrics as metrics

from common import Multi, Sentence
from word_vector_edit_distance import WVEditDistance, TunedNormalizedLevenshtein
from bipartite_graph_similarity import BipartiteGraphSim
from tree_edit_distance import TreeEditDistance


def two_class_data_iter(which):
    assert which in ["dev", "test", "train"]

    base_path = Path("/home/nrg/datasets/snli_1.0/snli_1.0")

    f = base_path / f"snli_1.0_{which}.jsonl"

    with f.open() as i:
        for line in i:
            loaded = json.loads(line)
            if loaded["gold_label"] == "-":
                continue
            yield (
                (Sentence(loaded["sentence1"], loaded["sentence1_parse"]), Sentence(loaded["sentence2"], loaded["sentence2_parse"])),
                loaded["gold_label"] == "entailment"
            )


def three_class_data_iter(which):
    assert which in ["dev", "test", "train"]

    base_path = Path("/home/nrg/datasets/snli_1.0/snli_1.0")

    f = base_path / f"snli_1.0_{which}.jsonl"

    with f.open() as i:
        for line in i:
            loaded = json.loads(line)
            if loaded["gold_label"] == "-":
                continue
            yield (
                (Sentence(loaded["sentence1"], loaded["sentence1_parse"]), Sentence(loaded["sentence2"], loaded["sentence2_parse"])),
                loaded["gold_label"]
            )


def evaluate(model, which="dev"):
    X, y = [], []

    for (s1, s2), label in three_class_data_iter(which):
        d = model.distance(s1, s2)
        #print(d)
        X.append(d)
        y.append(label)

    print(Counter(y))

    scores = sel.cross_validate(
        lm.LogisticRegression(),
        X,
        y=y,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cv=sel.StratifiedShuffleSplit(n_splits=5),
    )

    print(scores)


def main():
    evaluate(Multi([BipartiteGraphSim.load(), TunedNormalizedLevenshtein(del_cost=.5)]))


    # for _ in range(10):
    #     wved = BipartiteGraphSim.load_random()
    #     print(wved)
    #     evaluate(wved)

if __name__ == "__main__":
    main()
