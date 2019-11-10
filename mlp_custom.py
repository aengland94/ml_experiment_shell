# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:41:05 2019

@author: Anthony
"""
import numpy as np

class MLPCustom(object):
    """
    """
    def __init__(self, hidden_layer_sizes=(100, ), learning_rate_init=0.001, 
                 activation="logistic", max_iter=200):
        """
        Initalizes the DecisionTreeCustom
        :param hidden_layer_sizes: tuple of desired hidden layer sizes
        :param learning_rate_init: float value of learning rate
        :param activation: string activation key for the activation function 
        that is to be used ie 'logistic', 'relu', 'leaky relu', 'tanh', and 
        'identity'
        """
        self.n_layers = 2
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.weights = []
        self.bias = np.array([-1])
        self.bias_weights = []
        self.a = []
        self.errors = []
        self.targets = None
        
        self.n_layers += len(self.hidden_layer_sizes)
        
        if activation == "logistic":
            self.activation = self.logistic
            self.d_activation = self.d_logistic
        elif activation == "relu":
            self.activation = self.relu
            self.d_activation = self.d_relu
        elif activation == "leaky relu":
            self.activation = self.leaky_relu
            self.d_activation = self.d_leaky_relu
        elif activation == "tanh":
            self.activation = self.tanh
            self.d_activation = self.d_tanh
        else:
            self.activation = self.identity
            self.d_activation = self.d_identity
        pass
    
    
    def fit(self, data_train, targets_train):
        """
        Trains the model based on train data and targets
        :param data_train: array of training data
        :param targets_train: array of training targets
        """
        self.targets = np.unique(targets_train)        
        self.set_arrays(data_train.shape[1])    
        size = len(data_train)
        order = [x for x in range(size)]
        
        # train weights on data_train
        for epoch in range(self.max_iter):
            # suffle the order for each epoch
            correct = 0
            np.random.shuffle(order)
            for i in order:
                predicted_target = self.predict_target(data_train[i], True)
#                print("Predicted Target", predicted_target)
#                print("True Target", targets_train[i])
                if predicted_target[0] == targets_train[i]:
                    correct += 1
                
                # back propigation
                # find expected output for each output node
                targets_i = np.zeros(self.targets.shape)
                targets_i[np.where(self.targets==targets_train[i])] = 1
                # calculate errors for output nodes
                self.errors[-1] = self.d_activation(self.a[-1]) \
                                * (self.a[-1] - targets_i)
                # calculate errors for hidden nodes
                for j in reversed(range(len(self.hidden_layer_sizes))):
                    self.errors[j] = self.d_activation(self.a[j]) \
                                   * np.dot(self.weights[j + 1].transpose(), 
                                            self.errors[j + 1])
                
                self.update_weights(data_train[i])
                #print(self.bias_weights)
            
            
            print(str(epoch) + "," + str(correct / size))
            #print(self.weights, "\n\n")
                
    
        return
    
    
    def set_arrays(self, size):
        """
        Sets self.weights, self.bias_weights, self.a, and self.errors
        self.weights and self.bias_weights are small random values
        self.a and self.errors are zeros
        """
        for ls in self.hidden_layer_sizes:
            self.weights.append(np.random.uniform(-1, 1, (ls, size)))
            self.bias_weights.append(np.random.uniform(-1, 1, ls))
            size = ls            
        self.weights.append(
                np.random.uniform(-1, 1, (self.targets.shape[0], size)))
        self.bias_weights.append(
                np.random.uniform(-1, 1, self.targets.shape[0]))
        
        self.a = np.zeros((len(self.hidden_layer_sizes) + 1,)).tolist()
        self.errors = np.zeros((len(self.hidden_layer_sizes) + 1,)).tolist()
        return
    
    
    def update_weights(self, data):
        """
        """
        delta_bias_weight = self.learning_rate_init * np.dot(
                self.errors[0].reshape((self.errors[0].shape[0], 1)), 
                self.bias)
        self.bias_weights[0] -= delta_bias_weight
#        self.bias_weights -= np.dot(
#                self.errors[0].reshape((self.errors[0].shape[0], 1)), 
#                self.bias)
        self.weights[0] -= np.dot(
                self.errors[0].reshape((self.errors[0].shape[0], 1)), 
                data.reshape(1, data.shape[0]))
        
        for i in range(1, len(self.hidden_layer_sizes)):
            delta_bias_weight = self.learning_rate_init * np.dot(
                    self.errors[i].reshape((self.errors[i].shape[0], 1)), 
                    self.bias)
            self.bias_weights[i] -= delta_bias_weight
            self.weights[i] -= np.dot(
                    self.errors[i].reshape((self.errors[i].shape[0], 1)), 
                    self.a[i - 1].reshape((1, self.a[i - 1].shape[0])))
        return
    
    
    def predict(self, data_test):
        """
        Predicts the targets from the data_test
        :param data_test: array of test data
        :return: list of predicted targets
        """
        targets_predicted = []
        
        for data in data_test:            
            targets_predicted.append(self.predict_target(data))
        
        return targets_predicted
    
    
    def predict_target(self, data, update_a=False):
        """
        Predict single target by running an individual through 
        the neural network
        :param data: the data the prediction is run on
        :param update_a: boolean if self.a should be updated
        :return: single predicted target value
        """
        inout = data
        for i in range(len(self.weights)):
            h = np.dot(self.weights[i], inout) \
              + np.dot(self.bias_weights[i].reshape(
                      (self.bias_weights[i].shape[0], 1)), self.bias)
            inout = self.activation(h)
            
            if update_a:
                self.a[i] = inout
        
        return self.targets[inout == np.max(inout)]
    
    
    def logistic(self, x):
        """
        logistic sigmoid activation function
        """
        return 1 / (1 + np.e**-x)
    
    
    def d_logistic(self, a):
        """
        derivative of logistic sigmoid activation function
        """
        return a * (1 - a)
    
    
    def identity(self, x):
        """
        identity activation function
        """
        #print("using identity activation function")
        return x
    
    
    def d_identity(self, a):
        """
        derivative of identity activation function
        """
        return np.zeros(a.shape) + 1
    
    
    def relu(self, x):
        """
        rectified linear unit activation function
        """
        # x if i > 0 else 0 for i in x
        # replace all values in the vector 'x' that 
        # are not greater than 0 with 0
        return np.where(np.greater(x, 0), x, 0)
    
    
    def d_relu(self, a):
        """
        derivative of rectified linear unit activation function
        """
        return np.where(np.greater(a, 0), 1, 0)
    
    
    def leaky_relu(self, x):
        """
        leaky rectified linear unit activation function
        """
        # x if i > 0 else 0 for i in x
        # replace all values in the vector 'x' that 
        # are not greater than 0 with 0
        return np.where(np.greater(x, 0), x, 0.01 * x)
    
    
    def d_leaky_relu(self, a):
        """
        derivative of leaky rectified linear unit activation function
        """
        return np.where(np.greater(a, 0), 1, 0.01)
    
    
    def tanh(self, x):
        """
        tanh activation function
        :param x: numpy array of h() values
        """
        return np.tanh(x)
    
    
    def d_tanh(self, a):
        """
        derivative of tanh activation function
        :param a: numpy array of activation values g(h())
        """
        return 1 - a**2
    


import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    """
    Test the class
    """
#    data = datasets.load_iris(True)
#    target = data[1]
#    data = data[0]
    
    data = pd.read_csv("../data/letters.csv")
    target = data.letter.values
    data = data.drop('letter', axis=1).values
    
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    
    
#    data = np.array(
#            [[1,1,1], [1,0,1], [0,1,1], [0,0,0], [1,1,0], [1,0,0], [0,1,0]])
#    target = np.array([0,1,1,0,1,1,0])
    
    data_train, data_test, targets_train, targets_test = train_test_split(
            data, target, test_size=.1)
    
    custom = MLPCustom((500, 150, 50), activation="logistic", max_iter=50, 
                       learning_rate_init=0.012)
    custom.fit(data_train, targets_train)
    
    #print(custom.weights)
    
    return 0


if __name__ == "__main__":
    main()