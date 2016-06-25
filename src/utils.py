#!/usr/bin/env python
#-*-coding:utf-8-*-
from sklearn import metrics
def get_precision_score(y_true, y_pred):
    """
    """
    cnt_of_data = y_true.shape[0]
    cnt_of_accurate = 0
    for item_true, item_pred in zip(y_true, y_pred):
        if item_true == item_pred:
            cnt_of_accurate += 1
    res = 1.0 * cnt_of_accurate / cnt_of_data
    return res

def get_auc_score(y_true, y_pred):
    """
    """
    return metrics.roc_auc_score(y_true, y_pred)

def get_recall_score(y_true, y_pred):
    """
    """
    return metrics.recall_score(y_true, y_pred)
