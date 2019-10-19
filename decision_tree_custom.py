# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:33:44 2019

@author: Anthony
"""

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeCustom(object):
    """
    A Classifier that uses the k nearest neighbors 
    """
    def __init__(self, max_depth=None, classifier=True):
        """
        Initalizes the DecisionTreeCustom
        """
        self.max_depth = max_depth
        self.classifier = classifier
        self.id3_tree = None
        pass
    
    
    def fit(self, data_train, targets_train):
        """
        Trains the model based on train data and targets
        :param data_train: array of training data
        :param targets_train: array of training targets
        """
        df = pd.DataFrame.from_records(data_train)
        columns = df.columns
        targets = pd.Series.from_array(targets_train, name="target")
        df = pd.concat([df, targets], axis=1)
        
        self.id3_tree = self.create_id3_tree(df, columns, targets.unique())
        
        return
    
    
    def predict(self, data_test):
        """
        Predicts the mode of the neighbors found in self.kdTree
        :param data_test: array of test data
        :return: list of predicted targets
        """
        targets_predicted = []
        
        for data in data_test:            
            targets_predicted.append(self.predictTarget(self.id3_tree, data))
        
        return targets_predicted
    
    
    def score(self, data_test, targets_test):
        """
        Calculates a score that shows how well the targets_test were 
        represented by the targets_predicted from self.predict(data_test)
        :param data_test: array of test data
        :param targets_test: array of test tragets
        :return: float of accuracy between 0 and 1
        """           
        size = len(targets_test)
        
        if size > 0:
            targets_predicted = self.predict(data_test)
            
            if self.classifier:
                num_correct = 0 
                for i in range(size):
                    if targets_test[i] == targets_predicted[i]:
                        num_correct += 1
                        
                return num_correct / size
            else:
                return r2_score(targets_test, targets_predicted)
        else:
            print("No targets to test against. Pick a smaller training portion.")
            return 0
    
    
    def create_id3_tree(self, df, columns, targets, parent=None, depth=0):
        """
        Recersively creates a ID3 tree
        :param df: pandas dataframe that contains columns in columns 
        it also has a target column
        :param targets: set of targets
        :param parent: (optional) parent ID3Node
        None when route node
        :param depth: (optional) uint depth down the tree
        0 when route node
        :return: head of ID3 tree
        """
        if len(df.target.unique()) == 1:
            return ID3Node(target=df.target[df.target.first_valid_index()], parent=parent)
        elif columns.empty or \
        (self.max_depth != None and depth >= self.max_depth):
            if self.classifier:
                return ID3Node(target=df.target.mode()[0], parent=parent)
            else:
                return ID3Node(target=np.mean(df.target), parent=parent)
        else:
            entropy = []
            for c in columns:
                entropy.append(self.get_column_entropy(c, df, targets))
            
            column = columns[np.argmin(entropy)]
            columns = columns.drop(column)
            
            id3_node = ID3Node(column, parent=parent)
            
            keys = df[column].unique()
            for k in keys:
                split = df[df[column] == k]
                id3_node.children[k] = self.create_id3_tree(
                                split, columns, targets, id3_node, depth + 1)
            
            return id3_node            
    
    
    def predictTarget(self, id3_tree, data):
        """
        Recersively predicts the target of data by traversing id3_tree
        :param id3_tree: the head node of the tree being traversed
        :param data: the data used to traverse the tree
        """
        if id3_tree.is_leaf:
            return id3_tree.target
        
#        print("feature:",id3_tree.feature)
#        print("data:",data)
#        print("value:",data[id3_tree.feature])
#        print("keys:",id3_tree.children.keys())
        
        try:
            split = id3_tree.children[data[id3_tree.feature]]
        except KeyError:
            # key wasn't in training data, so just pick the first one
            # TODO: use better logic (pick most common key at level)
            keys = id3_tree.children.keys()
            for k in keys:
                key = k
                break
            split = id3_tree.children[key]
        
        return self.predictTarget(split, data)
    
    
    def get_column_entropy(self, c, df, targets):
        """
        :param c: column name to get entropy of
        :param df: pandas dataframe the column is in
        it is expected to also have a targets column
        :param targets: set of target values
        :return: weighted entropy for the given column (c)
        """
        entropies = []
        sizes = []
        splits = df[c].unique()
        
        for i in splits:
            split = df[df[c]==i]
            entropy = 0
            size = split.shape[0]
            for t in targets:
                target = split[split.target==t]
                p = target.shape[0] / size
                entropy += self.get_entropy(p)
                pass
            
            entropies.append(entropy)
            sizes.append(size)
            
        entropy = 0
        for i in range(len(entropies)):
            entropy += sizes[i] * entropies[i]
        
        return entropy
    
    
    def get_entropy(self, p):
        """
        :param p: float portion of whole between 0 and 1
        :return: entropy from given portion (p)
        """
        if not p == 0:
            return -1 * p * np.log2(p)
        else:
            return 0
        
    
    

class ID3Node(object):
    """
    A Tree Node that holds both a feature or a target
    Used to create a KD-tree
    """
    def __init__(self, feature=None, target=None, parent=None):
        """
        Initializes ID3Node
        """
        self.feature = feature
        self.target = target
        self.parent = parent
        self.children = {}
        pass
    
    
    @property
    def is_leaf(self):
        """
        :return: boolean if the node is a leaf
        """
        return len(self.children) == 0
    
    @property
    def is_route(self):
        """
        :return: boolean if node is the route node of the tree
        """
        return self.parent == None
    



def main():
    """
    Used for testing DecisionTreeCustom class
    """
#    data = pd.DataFrame.from_records(np.array(
#            [["c","d","y"], ["c","s","y"], ["d","d","y"], ["d","s","n"], 
#             ["c","d","n"], ["c","s","n"], ["d","d","n"]]))
#    target = pd.Series.from_array(np.array(["l","h","h","l","h","h","l"]), 
#                                  name="target")
    data = pd.DataFrame.from_records(np.array(
            [[1,1,1], [1,0,1], [0,1,1], [0,0,0], 
             [1,1,0], [1,0,0], [0,1,0]]))
    target = pd.Series.from_array(np.array([0,1,1,0,1,1,0]), 
                                  name="target")
    df = pd.concat([data,target], axis=1)
    
    predictions = []
    predictions2 = []
    
    for r in range(df.shape[0]):
        test = df.query("index=="+str(r))
        test = test.drop("target", axis=1).values
        
        train = df.query("index!="+str(r))
        train_data = train.drop("target", axis=1).values
        train_targets = train.target.values.flatten()
        
        custom = DecisionTreeCustom()
        custom.fit(train_data, train_targets)
        predictions.append(custom.predict(test)[0])
        
        tree = DecisionTreeClassifier()
        tree.fit(train_data, train_targets)
        predictions2.append(tree.predict(test)[0])
        
    print("true:", target.values)
    print("predicted:", predictions)
    print("sklearn:", predictions2)
    print("7-fold cross validation of toy data")
    print("Average Score:", accuracy_score(target.values, predictions))
    print("sklearn Average Score:", accuracy_score(target.values, predictions2))
    
    return 0


if __name__ == "__main__":
    main()