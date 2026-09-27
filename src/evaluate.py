"""End-to-end metrics include truth pairs lost by candidate generation."""
import numpy as np


def metrics(labels, probabilities, groups, truth_counts, threshold):
    labels, probabilities, groups = np.asarray(labels), np.asarray(probabilities), np.asarray(groups)
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    total = int(np.sum(truth_counts))
    fn = total-tp
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/total if total else 0.
    f = 1.25*tp/(1.25*tp+fp+.25*fn) if tp+fp+fn else 0.
    pred_counts = np.bincount(groups[predicted], minlength=len(truth_counts))
    singleton = np.asarray(truth_counts)==0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "F0.5": f,
            "true_positives":tp, "false_positives":fp, "false_negatives":fn,
            "candidate_recall": float(np.sum(labels)/total) if total else 0.,
            "singleton_accuracy": float(np.mean(pred_counts[singleton]==0)) if singleton.any() else None,
            "singleton_count": int(singleton.sum()), "true_pairs": total,
            "candidate_count":len(labels), "s1_count":len(truth_counts)}


def sweep(labels, probabilities, groups, truth_counts):
    thresholds = np.unique(np.r_[np.arange(.05,1,.025), .99, .995, .999])
    return [metrics(labels, probabilities, groups, truth_counts, t) for t in thresholds]
