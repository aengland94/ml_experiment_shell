# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:08:40 2019

@author: Anthony
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import numpy as np
import pandas as pd

from hard_coded_classifier import HardCodedClassifier
from knn_custom import KNNCustom
from decision_tree_custom import DecisionTreeCustom

class Experiment():
    """
    """
    def __init__(self):
        pass
    
    def run(self,experiment_val="", knn_k_num=-1, data_val="", 
            test_portion_num=-1, k_fold_num=-1):
        run_experiment(experiment_val, knn_k_num, data_val, test_portion_num, 
                       k_fold_num)



def main():
    """
    Displays welcome banner
    allows user to run experiments in a loop
    """
    display_welcome()
    
    running = True
    while running:
        run_experiment()
        
        run_another = input("Do you want to run another experiment? (y/n):\n")
        run_another = run_another.lower()
        
        if not (run_another == "y" or run_another == "yes"):
            running = False
        
    return 0


def run_experiment(experiment_val="", knn_k_num=-1, data_val="", 
                   test_portion_num=-1, k_fold_num=-1):
    """
    Conducts the chosen experiment
    :param experiment_val: (optional) string which experiment should be ran?
    :param knn_k_num: (optional) int what k for knn experiment should be used?
    :param data_val: (optional) string which data should be used?
    :param test_portion_num: (optional) int what portion of the data should be 
    used for testing?
    :param k_fold_num: (optional) int what k fold number should be used?
    """    
    model_y_vectors = choose_experiment(experiment_val, knn_k_num)
    
    data, target = load_data(data_val)
    
    test_portion = get_test_portion(test_portion_num)
    
    k = get_k_fold(k_fold_num)
    
    print("\nRunning {}-fold experiement...".format(k))
    
    for i in range(k):    
        data_train, data_test, targets_train, targets_test = \
            train_test_split(data, target, test_size=test_portion)      
        
        for m in model_y_vectors:
            test_model(m, data_train, data_test, targets_train, targets_test)
        
        print(".", end="")
    
    print()
            
    for model_y_vector in model_y_vectors:
        model = model_y_vector[0]
        y_vector = model_y_vector[1:]
        
        print("\nResults of {}-fold experiement\n".format(k) + 
              "on {}".format(type(model)))
        print("Avrg. Score: {}".format(round(np.mean(y_vector), 4)))
        
        
    return


def display_welcome():
    """
    Displays the welcome banner
    """
    print("\n\n" + 
          "|_     _|  | ____|  | |     | ___|  | ___ |   _|_ _|_   | ____|\n" + 
          "|_     _|  | |__    | |     | |     | | | |  _| |_| |_  | |__  \n" +
          "|_     _|  | ___|   | |     | |     | | | |  _|     |_  | ___| \n" +
          "|_ _|_ _|  | |___   | |__   | |__   | |_| |  _|     |_  | |___ \n" + 
          " |_| |_|   |_____|  |____|  |____|  |_____|  _|     |_  |_____|\n" +
          "\n" +
          "                  to your experiment shell!!!                 \n\n")
    return


def choose_experiment(experiment_val="", knn_k_num=-1, max_depth=None):
    """
    Has user choose an experiment to run
    :param experiment_val: (optional) string which experiment should be ran?
    :param knn_k_num: (optional) int what k for knn experiment should be used?
    :return: a list of model vectors
    """    
    if experiment_val == "":
        experiment = input("Type the number of your desired experiment:\n" + 
                           "\t1. Gaussian Naive Bayes vs Hard-coded Classifier\n" +
                           "\t2. K Neighbors Classifier vs KNN Custom\n" + 
                           "\t3. K Neighbors Regressor vs KNN Custom\n" + 
                           "\t4. Decision Tree Classifier vs Decision Tree Custom\n" + 
                           "\t5. Decision Tree Regressor vs Decision Tree Custom\n")
    else:
        experiment = experiment_val
    
    # get k for any KNN experiments
    if experiment in ["2", "3"]:
        if knn_k_num == -1:
            k = input("What k value would you like for this experiment?\n" + 
                      "(Possitive Whole Number):\n")
        else:
            k = knn_k_num
        try:
            k = int(float(k))
            if k < 1:
                k = 1
        except ValueError:
            k = 1
    
    # get max_depth for any Decision Tree experiments
    if experiment in ["4", "5"]:
        if max_depth == None:
            max_depth = input("What max depth would you like for your tree?\n" + 
                              "(Possitive Whole Number)(anything else for 'None':\n")
        try:
            max_depth = int(float(max_depth))
            if max_depth < 1:
                max_depth = None
        except ValueError:
            max_depth = None
            
    if experiment == "2":
        models = [[KNeighborsClassifier(k)], [KNNCustom(k)]]
    elif experiment == "3":
        models = [[KNeighborsRegressor(k)], [KNNCustom(k, classifier=False)]]
    elif experiment == "4":
        models = [[DecisionTreeClassifier(max_depth=max_depth)], 
                   [DecisionTreeCustom(max_depth=max_depth)]]
    elif experiment == "5":
        models = [[DecisionTreeRegressor(max_depth=max_depth)], 
                   [DecisionTreeCustom(classifier=False, max_depth=max_depth)]]
    else:
        models = [[GaussianNB()], [HardCodedClassifier()]]
        
    return models

    
def load_data(data_val=""):
    """
    Loads the data for the experiment
    :param data_val: (optional) string which data should be used?
    :return: the data that was loaded and its target
    """    
    if data_val == "":
        option = input("Type the number of the data you would like to use:\n" + 
                       "\t1. Iris (Classification) (continuous)\n" + 
                       "\t2. Car (Classification) (discrete)\n" + 
                       "\t3. Auto MPG (Regression) (mixed)\n" + 
                       "\t4. Student Math (Regression) (mixed)\n" + 
                       "\t5. Student Math (Regression) (discrete)\n")
    else:
        option = data_val
    
    if option == "2":
        data, target = clean_load_car_data()
    elif option == "3":
        data, target = clean_load_mpg_data()
    elif option == "4":
        data, target = clean_load_math_data()
    elif option == "5":
        data, target = clean_load_math_data_discrete()
    else:
        data = datasets.load_iris(True)
        target = data[1]
        data = data[0]
    
    return data, target


def clean_load_car_data():
    """
    Loads the car data and cleans it in prep for an experiement
    :return: cleaned car data and target
    """
    source = "https://archive.ics.uci.edu/ml/machine-learning-databases/" + \
        "car/car.data"
    # the target column is the car's acceptability
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safty", 
               "target"]
    car = pd.read_csv(source, header=None, names=columns)
    #car.describe()
    car = pd.get_dummies(car, prefix="doors", columns=["doors"])
    car = pd.get_dummies(car, prefix="persons", columns=["persons"])
    
    car.buying = car.buying.astype("category").map(
            {"vhigh":4, "high":3, "med":2, "low":1})
    car.maint = car.maint.astype("category").map(
            {"vhigh":4, "high":3, "med":2, "low":1})
    car.lug_boot = car.lug_boot.astype("category").map(
            {"big":3, "med":2, "small":1})
    car.safty = car.safty.astype("category").map(
            {"high":3, "med":2, "low":1})
    car.target = car.target.astype("category").cat.codes
    
    # TODO: standarize data
    
    target = car.target.as_matrix().flatten()
    data = car.drop("target", axis=1).as_matrix()
    
    return data, target


def clean_load_mpg_data():
    """
    Loads the mpg data and cleans it in prep for an experiement
    :return: cleaned mpg data and target
    """
    source = "https://archive.ics.uci.edu/ml/machine-learning-databases/" + \
        "auto-mpg/auto-mpg.data"
    # the target column is the mpg
    columns = ["target", "cylinders", "displacement", "horsepower", "weight", 
               "acceleration", "model_year", "origin", "car_name"]
    
    mpg = pd.read_csv(source, header=None, names=columns, delim_whitespace=True, na_values=["?"])
    #mpg.isnull().any()
    #mpg[mpg.isnull().any(axis=1)]
    # TODO: clean and standarize data
    
    # splitting car_name into make and model
    name_split = [x.split(" ", 1) for x in mpg.car_name]
    make = pd.Series([x[0] for x in name_split], name="make")  
#    model = []
#    for x in name_split:
#        if len(x) == 2:
#            model.append(x[1])
#        else:
#            model.append(np.NaN)
#    model = pd.Series(model, name="model")
    # add make and model to mpg and drop car_name
#    mpg = pd.concat([mpg, make, model], axis=1)
    mpg = pd.concat([mpg, make], axis=1)
    mpg = mpg.drop("car_name", axis=1)
    
    # cleaning make
    mpg.make = mpg.make.replace("vokswagen", "vw")
    mpg.make = mpg.make.replace("vw", "volkswagen")
    mpg.make = mpg.make.replace("toyouta", "toyota")
    mpg.make = mpg.make.replace("maxda", "mazda")
    mpg.make = mpg.make.replace("chevy", "chevrolet")
    mpg.make = mpg.make.replace("chevroelt", "chevrolet")
    mpg.make = mpg.make.replace("mercedes", "mercedes-benz")
    
    # dropping all rows with NaN for now because of time
    # TODO: take care of NaN in a better way
    mpg = mpg.dropna()
    
    mpg.make = mpg.make.astype("category").cat.codes
    
    target = mpg.target.as_matrix().flatten()
    data = mpg.drop("target", axis=1).as_matrix()
    
    return data, target


def clean_load_math_data():
    """
    Loads the math data and cleans it in prep for an experiement
    :return: cleaned math data and target
    """
    # This uses the student-mat data. It can be downloaded from the url below:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00320/
    source = "../data/student/student-mat.csv"
    # the target column is g3 which is the final grade
    columns = ["school", "female", "age", "urban", "famsize_more_than_3",  
               "parents_together", "m_edu", "f_edu", "m_job", "f_job", "reason",  
               "guadian", "traveltime", "studytime", "failures", "schoolsup",  
               "famsup", "paid", "activities", "nursery", "higher", "internet", 
               "romantic", "famrel", "freetime", "goout", "d_alc", "w_alc", 
               "health", "absences", "g1", "g2", "target"]
    math = pd.read_csv(source, header=None, names=columns, sep=";",skiprows=[0])
    
    # TODO: clean and standarize data
    #math.isnull().all()
    
    math.school = math.school.astype("category").map({"GP":1, "MS":0})
    math.female = math.female.astype("category").map({"F":1, "M":0})
    math.urban = math.urban.astype("category").map({"U":1, "R":0})
    math.famsize_more_than_3 = math.famsize_more_than_3.astype("category").map(
            {"GT3":1, "LE3":0})
    math.parents_together = math.parents_together.astype("category").map(
            {"T":1, "A":0})
    
    math.m_job = math.m_job.astype("category").cat.codes
    math.f_job = math.f_job.astype("category").cat.codes
    math.reason = math.reason.astype("category").cat.codes
    math.guadian = math.guadian.astype("category").cat.codes
    
    math.schoolsup = math.schoolsup.astype("category").map({"yes":1, "no":0})
    math.famsup = math.famsup.astype("category").map({"yes":1, "no":0})
    math.paid = math.paid.astype("category").map({"yes":1, "no":0})
    math.activities = math.activities.astype("category").map({"yes":1, "no":0})
    math.nursery = math.nursery.astype("category").map({"yes":1, "no":0})
    math.higher = math.higher.astype("category").map({"yes":1, "no":0})
    math.internet = math.internet.astype("category").map({"yes":1, "no":0})
    math.romantic = math.romantic.astype("category").map({"yes":1, "no":0})
    
    target = math.target.as_matrix().flatten()
    data = math.drop("target", axis=1).as_matrix()
    
    return data, target


def clean_load_math_data_discrete():
    """
    Loads the math data and cleans it in prep for an experiement
    makes sure all feature have discrete values
    :return: cleaned math data and target
    """
    # This uses the student-mat data. It can be downloaded from the url below:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00320/
    source = "../data/student/student-mat.csv"
    # the target column is g3 which is the final grade
    columns = ["school", "female", "age", "urban", "famsize_more_than_3",  
               "parents_together", "m_edu", "f_edu", "m_job", "f_job", "reason",  
               "guadian", "traveltime", "studytime", "failures", "schoolsup",  
               "famsup", "paid", "activities", "nursery", "higher", "internet", 
               "romantic", "famrel", "freetime", "goout", "d_alc", "w_alc", 
               "health", "has_absences", "g1_less_than_11", "g2_less_than_11", 
               "target"]
    math = pd.read_csv(source, header=None, names=columns, sep=";",skiprows=[0])
    
    # map text values into numbers    
    math.school = math.school.astype("category").map({"GP":1, "MS":0})
    math.female = math.female.astype("category").map({"F":1, "M":0})
    math.urban = math.urban.astype("category").map({"U":1, "R":0})
    math.famsize_more_than_3 = math.famsize_more_than_3.astype("category").map(
            {"GT3":1, "LE3":0})
    math.parents_together = math.parents_together.astype("category").map(
            {"T":1, "A":0})
    
    math.m_job = math.m_job.astype("category").cat.codes
    math.f_job = math.f_job.astype("category").cat.codes
    math.reason = math.reason.astype("category").cat.codes
    math.guadian = math.guadian.astype("category").cat.codes
    
    math.schoolsup = math.schoolsup.astype("category").map({"yes":1, "no":0})
    math.famsup = math.famsup.astype("category").map({"yes":1, "no":0})
    math.paid = math.paid.astype("category").map({"yes":1, "no":0})
    math.activities = math.activities.astype("category").map({"yes":1, "no":0})
    math.nursery = math.nursery.astype("category").map({"yes":1, "no":0})
    math.higher = math.higher.astype("category").map({"yes":1, "no":0})
    math.internet = math.internet.astype("category").map({"yes":1, "no":0})
    math.romantic = math.romantic.astype("category").map({"yes":1, "no":0})
    
    # make continuous data discrete
    math.has_absences = math.has_absences.replace(
            math.has_absences[math.has_absences > 0], 1)
    
    math.g1_less_than_11 = math.g1_less_than_11.replace(
            math.g1_less_than_11[math.g1_less_than_11 < 11], 1)
    math.g1_less_than_11 = math.g1_less_than_11.replace(
            math.g1_less_than_11[math.g1_less_than_11 >= 11], 0)
    
    math.g2_less_than_11 = math.g2_less_than_11.replace(
            math.g2_less_than_11[math.g2_less_than_11 < 11], 1)
    math.g2_less_than_11 = math.g2_less_than_11.replace(
            math.g2_less_than_11[math.g2_less_than_11 >= 11], 0)
    
    target = math.target.as_matrix().flatten()
    data = math.drop("target", axis=1).as_matrix()
    
    return data, target

    
def get_test_portion(test_portion_num=-1):
    """
    Props user for test portion
    :param test_portion_num: (optional) int what portion of the data should be 
    used for testing?
    :return: float test portion
    """
    if test_portion_num == -1:
        test_portion = input("Desired test portion? (Number between 0.0 and 1.0):\n")
    else:
        test_portion = test_portion_num
    try:
        test_portion = float(test_portion)
    except ValueError:
        test_portion = 0.3
        print("Invalid test portion. Test portion will be", test_portion)
    
    if round(test_portion, 2) >= 0.99:
        test_portion = 0.8
        print("Invalid test portion. Test portion will be", test_portion)
    elif test_portion <= 0.0:
        test_portion = 0.2
        print("Invalid test portion. Test portion will be", test_portion)
        
    return test_portion


def get_k_fold(k_fold_num=-1):
    """
    Gets value of k for k-fold experiement from user
    :param k_fold_num: (optional) int what k fold number should be used?
    Validates k
    :return: int k
    """
    if k_fold_num == -1:
        k = input("K for K-Fold Experient? (Possitive Whole Number):\n")
    else:
        k = k_fold_num
    try:
        k = int(k)
    except ValueError:
        k = 1
        
    if k < 1:
        k = 1
        
    return k


def test_model(model_y_vector, data_train, data_test, targets_train, targets_test):
    """
    Fits model to training data
    Tests model on testing data
    Appends model accuracy to model_y_vector
    :param model_y_vector: a vector that has the model to use in index 0 and 
    the accuracy of past predicted y values as the other indexes
    :param data_train: list of feature vectors for training model
    :param data_test: list of feature vectors for testing model
    :param targets_train: vector of targets that go with each vector in
    data_train
    :param targets_test: vector of targets that go with each vector in 
    data_test
    """
    model = model_y_vector[0]
    
    model.fit(data_train, targets_train)
#    targets_predicted = model.predict(data_test)
    # Now predicting and testing with the model's score method
    model_y_vector.append(model.score(data_test, targets_test))
    

if __name__ == "__main__":
    main()