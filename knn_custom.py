# -*- coding: utf-8 -*-
"""
Created on Fri Oct 04 18:57:03 2019

@author: Anthony
"""
import numpy as np
import statistics as stat

from sklearn.metrics import r2_score

class KNNCustom(object):
    """
    A Classifier that uses the k nearest neighbors 
    """
    def __init__(self, k=1, classifier=True):
        """
        Initalizes the KNNCustom
        """
        self.k = k
        self.classifier = classifier
        self.max_k = 1
        self.kdTree = None
        pass
    
    
    def fit(self, data_train, targets_train):
        """
        Trains the model based on train data and targets
        :param data_train: array of training data
        :param targets_train: array of training targets
        """
        self.kdTree = self.createKDTree(data_train, targets_train)
        self.max_k = len(data_train)
        
        return
    
    
    def predict(self, data_test):
        """
        Predicts the mode of the neighbors found in self.kdTree
        :param data_test: array of test data
        :return: list of predicted targets
        """
        targets_predicted = []
        
        if self.k > self.max_k:
            self.k = self.max_k
        
        for data in data_test:
            neighbors = []
            distances = []
            self.findNeighbors(self.kdTree, data, neighbors, distances)
            
            targets = [x.target for x in neighbors]
            if self.classifier:
                try:
                    targets_predicted.append(stat.mode(targets))
                except stat.StatisticsError:
                    targets_predicted.append(targets[np.argmin(distances)])
                    #print("used stat exception")
            else:
                targets_predicted.append(np.mean(targets))
        
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
    
    
    def createKDTree(self, data, targets, level=0, parent=None):
        """
        Recursively creates KD-tree
        A KD-tree is a BST that is split at the median each time
        :param data: array of data that will be put in the tree
        :param targets: array of targets that go with the data
        :param level: how far into the tree are we?
        :param parent: parent KDNode (None for route)
        :return: route of the tree
        """
        # check for early return
        data_len = len(data)
        if data_len < 1:
            return None
        elif data_len <=2:
            kdNode = KDNode(data[0], targets[0], parent)
            if data_len == 2:
                kdNode.right = KDNode(data[1], targets[1], kdNode)
            return kdNode
        
        # sort data and targets
        sorting_dimension = level % data.shape[1]
        indices = np.argsort(data[:,sorting_dimension])
        sorted_data = data[indices]
        sorted_targets = targets[indices]
        
        # split data and targets
        median_i = (data_len - 1) // 2
        left_data = sorted_data[:median_i,:]
        left_targets = sorted_targets[:median_i]
        right_data = sorted_data[median_i + 1:,:]
        right_targets = sorted_targets[median_i + 1:]
        
        # create new KDNode to return
        kdNode = KDNode(sorted_data[median_i,:],
                         sorted_targets[median_i],
                         parent)
        kdNode.left = self.createKDTree(left_data,
                                        left_targets,
                                        level + 1,
                                        kdNode)
        kdNode.right = self.createKDTree(right_data,
                                    right_targets,
                                    level + 1,
                                    kdNode)        
        return kdNode
    
    
    def findNeighbors(self, kdNode, data, neighbors, distances, level=0):
        """
        Recursively find the values for the neighbors and distances lists
        It looks for the self.k number of closest kdNodes
        :param kdNode: the KDNode in question as the nearest node to the data
        :param data: array of values being compared to kdNode
        :param neighbors: list of the estimated closest kdNodes
        It is used as an INOUT variable
        :param distances: list of the distances from data of each kdNode in
        neighbors. It is used as an INOUT variable
        :param level: how far into the KD-tree kdNode is
        :return:
        """
        # check for early return
        if kdNode.is_leaf:
            self._add_neighbor(kdNode, data, neighbors, distances)
            return
        
        sorting_dimension = level % data.shape[0]
        sibling = None
        
        # go until you can't go any more and check for sibling
        if data[sorting_dimension] < kdNode.data[sorting_dimension]:
            if not kdNode.left == None:
                self.findNeighbors(
                        kdNode.left, data, neighbors, distances, level + 1)
                if kdNode.has_2_children:
                    sibling = kdNode.right
        else:
            if not kdNode.right == None:
                self.findNeighbors(
                        kdNode.right, data, neighbors, distances, level + 1)
                if kdNode.has_2_children:
                    sibling = kdNode.left
        
        # test sibling
        if not sibling == None:
            self._add_neighbor(sibling, data, neighbors, distances)
            # test cousins
            if not sibling.left == None:
                self._add_neighbor(sibling.left, data, neighbors, distances)
            if not sibling.right == None:
                self._add_neighbor(sibling.right, data, neighbors, distances)
        
        # test this node
        self._add_neighbor(kdNode, data, neighbors, distances)
        
        return
    
    
    def _add_neighbor(self, kdNode, data, neighbors, distances):
        """
        Adds the kdNode to neighbors and its distance from data to distances 
        if it is valid
        :param kdNode: the KDNode of quetion
        :param data: a vector of data kdNode is compared to
        :param neighbors: list of kdNodes
        :param distances: list of float distances from data
        """
        distance = self.get_distance(kdNode.data, data)
        
        if len(neighbors) < self.k:
            neighbors.append(kdNode)
            distances.append(distance)
        elif distance < np.max(distances):
            max_i = np.argmax(distances)
            distances[max_i] = distance
            neighbors[max_i] = kdNode
        
        return
    
    
    def get_distance(self, a, b):
        """
        Gets the distance between two points
        :param a: first point
        :param b: secound point
        :return: float distance
        """
        return np.sum((a - b)**2)



class KDNode(object):
    """
    A BST Node that holds both data and a target
    Used to create a KD-tree
    """
    def __init__(self, data=None, target=None, parent=None):
        """
        Initializes KDNode
        """
        self.data = data
        self.target = target
        self.parent = parent
        self.left = None
        self.right = None
        pass
    
    
    @property
    def is_leaf(self):
        """
        :return: boolean if the node is a leaf
        """
        return self.left == None and self.right == None
    
    @property
    def is_route(self):
        """
        :return: boolean if node is the route node of the tree
        """
        return self.parent == None
    
    @property
    def has_2_children(self):
        """
        :return: boolean if the node has both a left and right child node
        """
        return not (self.left == None or self.right == None)
    
    @property
    def has_sibling(self):
        """
        :returns: boolean if node has a sibling
        """
        if not self.is_route:
            return self.parent.has_2_children
        
        return False
    