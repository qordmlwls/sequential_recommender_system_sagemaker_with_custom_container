import numpy as np


def calc_precision(topk_preds, target_array):
    """
    calculate accuracy
    :param topk_preds: predictions
    :param target_array: targets
    :param k: number of rank
    :return: metric
    """
    # metric = []

    tp_fp = len(set(list(target_array)))
    tp = len(set(list(topk_preds)) & set(list(target_array)))
    # tp, fp = 0, 0
    #
    # for pred in topk_preds:
    #     if pred in target_array:
    #         tp += 1
    #         metric.append(tp / (tp + fp))
    #     else:
    #         fp += 1
    # return np.sum(metric) / k
    try:
        return tp/tp_fp
    except ZeroDivisionError:
        return np.nan