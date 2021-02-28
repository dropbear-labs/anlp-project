import json
from pathlib import Path
from collections import Counter, defaultdict

import tabulate
import pandas as pd
import numpy as np
import more_itertools as mit
import tqdm
import nltk
import sklearn.model_selection as sel
import sklearn.linear_model as lm
import sklearn.metrics as metrics

from common import Sentence
from word_vector_edit_distance import WVEditDistance, TunedNormalizedLevenshtein, CharBasedTunedLevenshtein, WVEditDistanceV2
from bipartite_graph_similarity import BipartiteGraphSim, TunedBipartiteGraphSim, FreqWeightedBipartiteGraphSim
from tree_edit_distance import TreeEditDistance
from string_distance import LongestCommonSubsequence, Bakkelund, LengthDiff, LengthSum


base_path = Path("/home/nrg/datasets/snli_1.0/snli_1.0")


def three_class_data_iter(which, n=None):
    assert which in ["dev", "test", "train"]

    f = base_path / f"snli_1.0_{which}.jsonl"

    with f.open() as i:
        yielded = 0
        for line in i:
            if n is not None and yielded >= n:
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
            yielded += 1


def evaluate(model, which="dev"):
    X, y = [], []

    for (s1, s2), label in three_class_data_iter(which):
        d = model.distance(s1, s2)
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


def label_tree_edit_distance(which, n=None):
    # precompute tree edit distance
    output = f"data/tree_edit_{which}.jsonl"
    model = TreeEditDistance()
    with open(output, "w") as o:
        for (s1, s2), label in tqdm.tqdm(three_class_data_iter(which, n=n)):
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


def train_full():
    sentences, labels = defaultdict(list), defaultdict(list)

    for (s1, s2), label in three_class_data_iter("train", n=100000):
        sentences["train"].append((s1, s2))
        labels["train"].append(label)
    for (s1, s2), label in three_class_data_iter("dev"):
        sentences["dev"].append((s1, s2))
        labels["dev"].append(label)
    for (s1, s2), label in three_class_data_iter("test"):
        sentences["test"].append((s1, s2))
        labels["test"].append(label)

    dfs = defaultdict(pd.DataFrame)

    # load tree edit distance from precomputed
    for which in ["train", "dev", "test"]:
        with open(f"data/tree_edit_{which}.jsonl") as i:
            dfs[which]["tree_edit_distance"] = [json.loads(line)[0] for line in i]

    # create parametrised features based on best performance on previous hyperparam opt runs
    features = {
        "wved": WVEditDistanceV2.load(**{"subs_cost": 1.902343873666979, "insert_cost": 1.3481990347842965, "del_cost": 0.07970709023722056}),
        "bipartite_graph_matching": BipartiteGraphSim.load(),
        "longest_common_subsequence": LongestCommonSubsequence(),
        "char_based_tuned_levenshtein": CharBasedTunedLevenshtein(**{"del_cost": 0.22665831415016702, "insert_cost": 0.4379537631692163, "subs_cost": 1, "normalize": True}),
        "length_diff": LengthDiff(),
        "length_sum": LengthSum()
    }

    for which in ["train", "dev", "test"]:
        for feature_name, feature in features.items():
            print(which, feature_name)
            dfs[which][feature_name] = [feature.distance(s1, s2)[0] for s1, s2 in sentences[which]]

    clf = xgboost.XGBClassifier(n_estimators=200, max_depth=5)
    fitted = clf.fit(dfs["train"], labels["train"])
    dev_acc = fitted.score(dfs["dev"], labels["dev"])
    test_acc = fitted.score(dfs["test"], labels["test"])

    feature_importances = dict(zip(dfs["train"].columns, fitted.feature_importances_))

    print(f"dev accuracy: {dev_acc}")
    print(f"test accuracy: {test_acc}")
    print(tabulate.tabulate(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)))


def generate_features_files():
    # for sharing with others
    sentences = defaultdict(list)
    for (s1, s2), label in three_class_data_iter("train", n=100000):
        sentences["train"].append((s1, s2))
    for (s1, s2), label in three_class_data_iter("dev"):
        sentences["dev"].append((s1, s2))
    for (s1, s2), label in three_class_data_iter("test"):
        sentences["test"].append((s1, s2))

    # create parametrised features based on best performance on previous hyperparam opt runs
    features = {
        "wved": WVEditDistanceV2.load(**{"subs_cost": 1.902343873666979, "insert_cost": 1.3481990347842965, "del_cost": 0.07970709023722056}),
        "bipartite_graph_matching": BipartiteGraphSim.load(),
        "longest_common_subsequence": LongestCommonSubsequence(),
        "char_based_tuned_levenshtein": CharBasedTunedLevenshtein(**{"del_cost": 0.22665831415016702, "insert_cost": 0.4379537631692163, "subs_cost": 1, "normalize": True}),
        "length_diff": LengthDiff(),
        "length_sum": LengthSum()
        # skip Tree
    }

    dfs = defaultdict(pd.DataFrame)

    for which in ["train", "dev", "test"]:
        for feature_name, feature in features.items():
            print(which, feature_name)
            dfs[which][feature_name] = [feature.distance(s1, s2)[0] for s1, s2 in sentences[which]]

    for which in ["train", "dev", "test"]:
        np.save(f"data/features_{which}.npz", dfs[which].to_numpy())


if __name__ == "__main__":
    #hyperparams_test(WVEditDistance)
    #hyperparams_test(TunedNormalizedLevenshtein)
    hyperparams_test(BipartiteGraphSim, n=1)
    #hyperparams_test(WVEditDistanceV2, n=10)
    #hyperparams_test(LongestCommonSubsequence, n=1)
    #hyperparams_test(Bakkelund, n=1)
    #hyperparams_test(FreqWeightedBipartiteGraphSim, n=1)
    #FreqWeightedBipartiteGraphSim


    #label_tree_edit_distance("dev")
    #tree_edit_distance_eval()

    #label_tree_edit_distance("test", n=100000)
