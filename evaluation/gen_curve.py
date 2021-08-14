from collections import defaultdict
from itertools import chain
from typing import NamedTuple, Tuple

import numpy as np


class Curve(NamedTuple):
    precision_points: Tuple[str]
    recall_points: Tuple[str]
    f1_points: Tuple[str]
    threshold_points: Tuple[str]


def p_r_f(tp, fp, fn):
    if tp == 0:
        if fp == 0 and fn == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1


def micro_p_r_f(tp_fp_fn_list):
    TP = sum(x[0] for x in tp_fp_fn_list)
    FP = sum(x[1] for x in tp_fp_fn_list)
    FN = sum(x[2] for x in tp_fp_fn_list)
    p, r, f = p_r_f(TP, FP, FN)
    return TP, FP, FN, p, r, f


def macro_p_r_f(tp_fp_fn_list):
    i = 0
    Precision = 0.0
    Recall = 0.0
    F1 = 0.0
    for tp, fp, fn in tp_fp_fn_list:
        precision, recall, f1 = p_r_f(tp, fp, fn)
        Precision += precision
        Recall += recall
        F1 += f1
        i += 1
    return Precision / i, Recall / i, F1 / i


def compute_metrics(preds, golds):
    metrics = defaultdict(float)

    num_datapoints = len(golds)
    assert len(preds) == num_datapoints

    for g, p in zip(golds, preds):
        g_labels = set(g)
        p_labels = set(p)
        inter = p_labels.intersection(g_labels)

        metrics["tp"] += len(inter)
        metrics["fp"] += len(p_labels.difference(g_labels))
        metrics["fn"] += len(g_labels.difference(p_labels))

        for k in [1, 3, 5]:
            topk_inter = set(p[:k]).intersection(g_labels)
            metrics[f"P@{k}"] += (1.0) * len(topk_inter) / k

        for k in [1, 3, 5]:
            metrics[f"P@{k}"] = 1.0 / num_datapoints * metrics[f"P@{k}"]

    metrics["micro_precision"], metrics["micro_recall"], metrics["micro_f1"] = p_r_f(
        metrics["tp"], metrics["fp"], metrics["fn"]
    )

    return metrics


def filter_labels(labels, filter_func):
    """
    Apply filter_func to the labels sequences of all samples, return filtered labels.

    Args:
        labels: type: list of list. Each list in labels is the raw decoded labels with scores
            Specifically, labels can be in shape data_size * [ example_id, Dict of {label:score}],
        filter_func: a function applied on list
        labels_f: similar shape to the labels. data_size * [ example_id, Dict of {label:score}]
        sequences of filtered labels for all samples.

    """
    labels_f = []
    for _, labels_per_sample in labels:  # (id , {label:score,...})
        labels_filtered = list(
            filter(filter_func, labels_per_sample.items())
        )  # e.g filter_func: lambda x: x[1] >= threshold
        labels_f.append([k for k, v in labels_filtered])
    return labels_f


def gen_curve(
    predictions,
    targets,
    num_thresholds: int = 50,
    min_support: float = 0,
):
    """
    Return a dictionary recording paired performance metrics computed by varying a an attribute
    (i.e. score). For now, it can be marginal probability in beam search. Note that
    min_support helps to remove unreliable estimates at high thresholds.

    Args:
        predictions: predicted labels sequences of all the samples. One line per sample. Each label comes with its score.
        targets: ground truth labels sequences of all the samples. One line per sample.
        num_thresholds: number of points to scatter between minimum score and maximum score.
        min_support: the minimum threshold of fp + tp to consider metrics as valid to record.

    metrics_all includes:
        "threshold_points"
        "micro_precision"
        "micro_recall"
        "micro_f1"
        "p@1", "p@3", "p@5"
        "fp", "tp", "fn"

    """

    # metrics_all is the record dictionary to keep all the metrics and the result to return
    metrics_all = defaultdict(list)

    scores_sep = [pred[1].values() for pred in predictions]
    scores = list(chain(*scores_sep))
    if len(scores) == 0:
        # Can't compute curve on an empty subset
        return None

    for threshold in np.linspace(
        min(scores), max(scores), num_thresholds
    ):  # return an ndarray of shape (num_thresholds, )
        # predictions : size * 2 (id , Dict of {label:score})
        t_predictions = filter_labels(  # entities -> labels
            predictions, lambda x: x[1] >= threshold
        )
        metric = compute_metrics(t_predictions, targets)
        if metric["tp"] + metric["fp"] > min_support:
            for k, v in metric.items():
                metrics_all[k].append(v)
            metrics_all["threshold_points"].append(threshold)

    # c1 = Curve(precision_points, recall_points, f1_points, threshold_points)
    c = Curve(
        metrics_all["micro_precision"],
        metrics_all["micro_recall"],
        metrics_all["micro_f1"],
        metrics_all["threshold_points"],
    )
    # TODO add plot in next task

    return metrics_all, c
