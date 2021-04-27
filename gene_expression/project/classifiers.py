#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:07:12 2021
These are few classifiers to be tried:  RandomForest or GradientBoostingClassifier 
@author: subashkhanal
"""

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)

def classifier(name,n_estimators=100):
    if name == 'RandomForestClassifier':
        return RandomForestClassifier(n_estimators=n_estimators,criterion='gini',class_weight='balanced')
    
    if name == 'GradientBoosting':
        return GradientBoostingClassifier(n_estimators=n_estimators)
    
    else:
        raise NotImplementedError("This classifier is not implemented yet. Please choose from [RandomForest, GradientBoosting]")
        