#!/usr/bin/env python
#-*-coding: utf-8-*-
import numpy as np
import scipy as sp
import pickle
import getopt
import sys
import utils

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.externals import joblib

def train_new_model(model_type,
                    model_params,
                    train_data_x,
                    train_data_y,
                    save_model_path=None):
    """
    Params
    ------
    model_type: the model name
                str

    model_params: params setting with corresponding model
                  alway from model_params_config file
                  dict

    train_data_x: the train x
                  numpy.ndarray
    train_data_y: the train y
                  numpy.ndarray

    save_model_path: if not None then save your model to the filepath afeter training finished
                     str or None

    Return
    ------
    trained_model: Generic Model Type from sklearn or None
    """
    trained_model = None
    if model_type == "logistic_regression":
        print "logistic regression model training"
        trained_model = LogisticRegression(
            C=model_params.get('C'),
            penalty=model_params.get('penalty'),
            solver=model_params.get('solver'),
            n_jobs=model_params.get('n_jobs')
        )
        trained_model.fit(train_data_x, train_data_y)
        print "model trained finished"
    ### CUSTOM MODIFICATION
    ### you can add any other model and corresponding params here!
    ###
    ###

    if trained_model != None and save_model_path != None:
        print "### SAVE MODEL to %s"%(save_model_path)
        joblib.dump(trained_model, save_model_path)
    return trained_model

def load_model(save_model_path):
    """
    load trained model directly from filepath

    Params
    ------
    save_model_path: the filepath of model trained
                     str

    Return
    ------
    trained_model: Generic Model Type from sklearn or None
    """
    trained_model = None
    try:
        trained_model = joblib.load(save_model_path)
    except:
        print "Fali to load model!"
    finally:
        return trained_model

def model_metrics(trained_model,
                  model_type,
                  test_data_x,
                  test_data_y,
                  save_result_path=None):
    """
    """
    y_true = None
    y_pred = None
    if model_type == "logistic_regression":
        y_pred = trained_model.predict(test_data_x)
        y_true = test_data_y

    ## result benchmark
    precision_score = utils.get_precision_score(y_true, y_pred)
    auc_score = utils.get_auc_score(y_true, y_pred)
    recall_score = utils.get_recall_score(y_true, y_pred)
    print "precision score: ", precision_score
    print "auc score: ", auc_score
    print "recall score: ", recall_score
    return
