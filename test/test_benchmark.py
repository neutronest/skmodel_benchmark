#!/usr/bin/env python
#-*-coding:utf-8-*-

import sys
import unittest2 as unittest
import numpy as np

sys.path.append("../src/")
import model_params_config
import benchmark

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn import datasets


class test_benchmark(unittest.TestCase):
    """
    """


    def pre_data(self):
        iris = datasets.load_iris()
        X = np.array([[0.1, 0.2],
                     [0.2, 0.5],
                     [-0.1, 0.33],
                     [0.3, 0.5],
                     [0.3, 0.23],
                     [0.134, 0.56],
                     [0.354, 0.657]])
        Y = np.array([1, 0, 1, 0, 1, 0, 1])
        self.train_data_x = X[:5, :]
        self.train_data_y = Y[:5,]
        self.test_data_x = X[5:, :]
        self.test_data_y = Y[5:,]

        #xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        #self.test_data_x = np.c_[xx.ravel(), yy.ravel()]
        #self.test_data_y = np.c_[yy.ravel()]
        return

    def test_train_model(self):
        """
        """
        self.pre_data()
        model = benchmark.train_new_model("logistic_regression",
                                          model_params_config.logistic_regression_params,
                                          self.train_data_x,
                                          self.train_data_y,
                                          "./test_tmp/test_logres_model"

        )
        print "[Test logistic regression model OK]"
        return model

    def test_model_metrics(self):
        """
        """
        self.pre_data()
        model = benchmark.load_model("./test_tmp/test_logres_model")
        if model == None:
            model = self.test_train_model()
        benchmark.model_metrics(model,
                                "logistic_regression",
                                self.test_data_x,
                                self.test_data_y)
        
        return



if __name__ == "__main__":
    unittest.main()
    
