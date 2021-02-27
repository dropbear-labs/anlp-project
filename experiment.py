import json
from pathlib import Path
from collections import Counter

import numpy as np
import more_itertools as mit
import tqdm
import nltk
import sklearn.model_selection as sel
import sklearn.linear_model as lm
import sklearn.metrics as metrics

from common import Sentence
from word_vector_edit_distance import WVEditDistance, TunedNormalizedLevenshtein, CharBasedTunedLevenshtein, WVEditDistanceV2
from bipartite_graph_similarity import BipartiteGraphSim, TunedBipartiteGraphSim
from tree_edit_distance import TreeEditDistance
from string_distance import LongestCommonSubsequence, Bakkelund


base_path = Path("/home/nrg/datasets/snli_1.0/snli_1.0")


def two_class_data_iter(which):
    assert which in ["dev", "test", "train"]

    f = base_path / f"snli_1.0_{which}.jsonl"

    with f.open() as i:
        for line in i:
            loaded = json.loads(line)
            if loaded["gold_label"] == "-":
                continue
            yield (
                (
                    Sentence(loaded["sentence1"], loaded["sentence1_parse"]),
                    Sentence(loaded["sentence2"], loaded["sentence2_parse"])
                ),
                loaded["gold_label"] == "entailment"
            )


def three_class_data_iter(which, n=None):
    assert which in ["dev", "test", "train"]

    f = base_path / f"snli_1.0_{which}.jsonl"

    with f.open() as i:
        for line_idx, line in enumerate(i, start=1):
            if n is not None and line_idx > n:
                break
            loaded = json.loads(line)
            if loaded["gold_label"] == "-":
                continue
            yield (
                (
                    Sentence(loaded["sentence1"], loaded["sentence1_parse"]),
                    Sentence(loaded["sentence2"], loaded["sentence2_parse"])
                ),
                loaded["gold_label"]
            )


def evaluate(model, which="dev"):
    X, y = [], []

    for (s1, s2), label in three_class_data_iter(which):
        d = model.distance(s1, s2)
        #print(d)
        X.append(d)
        y.append(label)

    scores = sel.cross_validate(
        lm.LogisticRegression(),
        X,
        y=y,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cv=sel.StratifiedShuffleSplit(n_splits=5),
    )

    return scores


def hyperparams_test(model, n=20):

    results = []

    for _ in tqdm.tqdm(range(n)):
        model_with_params = model.load_random()
        cv_scores = evaluate(model_with_params)["test_score"]
        results.append(
            {
                "params": model_with_params.params(),
                "cv_scores": [float(score) for score in cv_scores],
                "mean_score": float(np.mean(cv_scores))
            }
        )

    print(
        json.dumps(sorted(results, key=lambda x: x["mean_score"], reverse=True))
    )


def label_tree_edit_distance(which):
    output = f"data/tree_edit_{which}.jsonl"
    model = TreeEditDistance()
    with open(output, "w") as o:
        for (s1, s2), label in tqdm.tqdm(three_class_data_iter(which)):
            d = model.distance(s1, s2)
            print(str(d), file=o)


def tree_edit_distance_eval():
    which = "dev"
    data_f = "data/tree_edit_dev.jsonl"

    with open(data_f) as i:
        X = [json.loads(line) for line in i]

    y = [label for _, label in three_class_data_iter(which)]

    cv_scores = sel.cross_validate(
        lm.LogisticRegression(),
        X,
        y=y,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cv=sel.StratifiedShuffleSplit(n_splits=5),
    )["test_score"]

    print(
        json.dumps(
            {
                "cv_scores": [float(score) for score in cv_scores],
                "mean_score": float(np.mean(cv_scores))
            }
        )
    )



# def generate_test_train_multimodel():

#     models = [
#         WVEditDistance.load(),
#         TunedNormalizedLevenshtein(del_cost=.5),
#         BipartiteGraphSim.load(),
#         TreeEditDistance()
#     ]

#     Xs = []
#     ys = []

#     for (s1, s2), label in tqdm.tqdm(three_class_data_iter("train")):
#         ds = [model.distance(s1, s2) for model in models]
#         Xs.append(d)
#         ys.append(label)

#     results = []

#     for _ in tqdm.tqdm(range(20)):
#         wved = WVEditDistance.load_random()
#         cv_scores = evaluate(wved)["test_score"]
#         results.append(
#             {
#                 "params": wved.params(),
#                 "cv_scores": [float(score) for score in cv_scores],
#                 "mean_score": float(np.mean(cv_scores))
#             }
#         )

#     print(
#         json.dumps(sorted(results, key=lambda x: x["mean_score"], reverse=True))
#     )



if __name__ == "__main__":
    #hyperparams_test(WVEditDistance)
    #hyperparams_test(TunedNormalizedLevenshtein)
    #hyperparams_test(BipartiteGraphSim, n=1)
    #hyperparams_test(WVEditDistanceV2, n=10)
    #hyperparams_test(LongestCommonSubsequence, n=1)
    #hyperparams_test(Bakkelund, n=1)

    #label_tree_edit_distance("dev")
    #tree_edit_distance_eval()
