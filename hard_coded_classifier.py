# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:27:58 2019

@author: Anthony
"""

class HardCodedClassifier():
    """
    A Classifier that doesn't learn
    It predicts the same value every time
    """
    def __init__(self):
        """
        Initalizes the HardCodedClassifier
        Doesn't do anything right now
        """
        pass
    
    def fit(self, data_train, targets_train):
        """
        Trains the model based on train data and targets
        :param data_train: array of training data
        :param targets_train: array of training targets
        """
        pass
    
    def predict(self, data_test):
        """
        Predicts '0' for each target
        :param data_test: array of test data
        :return: list of predicted targets
        """
        targets_predicted = []
        
        for i in data_test:
            targets_predicted.append(0)
        
        return targets_predicted
    
    def score(self, data_test, targets_test):
        """
        Calculates a score that shows how well the targets_test were 
        represented by the targets_predicted from self.predict(data_test)
        :param data_test: array of test data
        :param targets_test: array of test tragets
        :return: float of accuracy between 0 and 1
        """
        num_correct = 0    
        size = len(targets_test)
        
        if size > 0:
            targets_predicted = self.predict(data_test)
            for i in range(size):
                if targets_test[i] == targets_predicted[i]:
                    num_correct += 1
                    
            return num_correct / size
        else:
            print("No targets to test against. Pick a smaller training portion.")
            return 0