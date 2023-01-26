import pickle as pickle
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def santiment_analysis_micro_f1(preds, labels):
    label_indices = [-1,0,1]
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def santiment_analysis_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(3)[labels]

    score = np.zeros((3,))
    for c in range(3):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = santiment_analysis_micro_f1(preds, labels)
    auprc = santiment_analysis_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def label_to_num(labels):
    num_label = {"negative": -1, "neutral" : 0, "positive" : 1}
    output = []
    for label in labels:
        output.append(num_label[label])
    return output


def num_to_label(labels):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = {-1 : "negative", 0 : "neutral", 1 : "positive"}
    output = []
    for label in labels:
        output.append(origin_label[label])
    return output
