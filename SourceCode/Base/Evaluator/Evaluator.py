"""
@author: Lim Yuan Her
"""

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

class Evaluator(object):

    def __init__(self):
        self.metrics = []

    def addMetric(self, metricName):
        if(type(metricName) == str):
            self.metrics.append(metricName)
        elif(type(metricName) == list):
            self.metrics = metricName

    def Evaluate(self, pred, actual, K=5):
        results = {}

        for metric in self.metrics:
            if(metric == "rmse"):
                results["rmse"] = self._get_rmse(pred, actual)
            elif(metric == "mapK"): 
                 results["mapK"] = self._get_mapK(pred, actual, K)
            elif(metric == "marK"): 
                results["marK"] = self._get_marK(pred, actual, K)
            elif(metric == "coverage"): 
                results["coverage"] = self._get_coverage(pred, actual, K)

        return results      

    def _get_rmse(self, pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        pred = np.nan_to_num(pred)
        actual = actual[actual.nonzero()].flatten()
        rmse = sqrt(mean_squared_error(pred, actual))
        return rmse

    def _get_mapK(self, pred, actual, K):
        # Get 
        pred_topK = [ sorted(range(len(rating)), key=lambda i: rating[i])[-K:] for rating in pred ]
        actual_rated = [ np.nonzero(rating)[0] for rating in actual ]

        prec = [ list(set(p).intersection(actual_rated[i])) for i,p in enumerate(pred_topK) ]
            
        mapK = np.mean([ np.round(float(len(p)) / float(len(pred_topK[i])), 4) for i,p in enumerate(prec) ])

        return mapK

    def _get_marK(self, pred, actual, K):
        # Get 
        pred_topK = [ sorted(range(len(rating)), key=lambda i: rating[i])[-K:] for rating in pred ]
        actual_rated = [ np.nonzero(rating)[0] for rating in actual ]

        rec = [ list(set(p).intersection(actual_rated[i])) for i,p in enumerate(pred_topK) ]
            
        marK = np.mean([ np.round(float(len(p)) / float(len(pactual_rated[i])), 4) for i,p in enumerate(rec) ])

        return marK

    def _get_coverage(self, pred, actual, K):
        pred_topK = [ sorted(range(len(rating)), key=lambda i: rating[i])[-K:] for rating in pred ]
        predicted_flattened = [p for sublist in pred_topK for p in sublist]
        unique_predictions = len(set(predicted_flattened))
        totals = pred.shape[1]
        coverage = round(unique_predictions/(totals* 1.0)*100,2)
        return coverage
