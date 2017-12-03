#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:00:34 2017

@author: austinlee
"""
from collections import Counter
from evaluation import Eval

class Perceptron:
    def __init__(self, train_docs, train_labels, MAX_ITERATIONS=100, dev_docs=None, dev_labels=None):
        self.CLASSES = ['AGAINST', 'FAVOR', 'NONE']
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        self.weights = {s: Counter() for s in self.CLASSES}
        self.learn(train_docs, train_labels)

    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {s: Counter(c) for s,c in self.weights.items()}

    def learn(self, train_docs, train_labels):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """
        # Initialize variables, weight_dict holds weight sums for each language, defined below
        docNum = 0
        
        #Set weight vector to zero
        for iteration in range(0, self.MAX_ITERATIONS):
            # Initialize variables for accuracy calculations per iteration
            docNum = 0
            updates = 0
            correctCount = 0
            correctCountDev = 0
            trainSize = len(train_docs)
            devSize = len(self.dev_docs)
            
            for doc in train_docs:
                # Find the stance with the greatest weight
                prediction = self.predict(doc)
                actual = train_labels[docNum]
                
                # Assess whether the prediction matches the actual
                # If it is a mistake, update weights
                if prediction != actual:
                    #print(actual)
                    updates+=1
                    # Increase weights of correct vector
                    for key in doc:
                        self.weights[actual][key] = self.weights[actual][key]+doc[key]
                        # Decrease weights of incorrect vectors
                        self.weights[prediction][key] = self.weights[prediction][key]-doc[key]
                else:
                    correctCount+=1
                    
                docNum+=1
                
            # Test on dev_docs
            docNum=0
            for doc in self.dev_docs:
                # Find the stance with the greatest weight
                dev_prediction = self.predict(doc)
                dev_actual = self.dev_labels[docNum]
                
                # Assess prediction
                if dev_prediction == dev_actual:
                    correctCountDev+=1
                docNum+=1
            
            print("iteration: " + str(iteration) + ", updates=" + str(updates) + ", trainAcc=" + str(correctCount/trainSize) + ", devAcc=" + str(correctCountDev/devSize))
            # Stop iterations if training data is fully separated
            if correctCount/trainSize == 1:
                break

    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """
        return ...

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        # Initialize variables, weight_dict holds weight sums for each language, defined below
        weight_dict = {}
        against_sum=0
        favor_sum=0
        none_sum=0
        
        for key in doc:
            against_sum = against_sum + (doc[key]*self.weights["AGAINST"][key])
            favor_sum = favor_sum + (doc[key]*self.weights["FAVOR"][key])
            none_sum = none_sum + (doc[key]*self.weights["NONE"][key])
        
        # Input weight sums into data structure for python max computing
        weight_dict["AGAINST"] = against_sum
        weight_dict["FAVOR"] = favor_sum
        weight_dict["NONE"] = none_sum
        
        
        # Find the native language with the greatest weight
        prediction = max(weight_dict, key=weight_dict.get)
        
        return prediction

    def printTen(self):
        for stance in self.CLASSES:
            ten_highest_keys=[]
            ten_highest_weights=[]
            maximum_weight=0
            maximum_key=""
            for x in range(0,10):
                maximum_key=max(self.weights[stance], key=self.weights[stance].get)
                maximum_weight=self.weights[stance][maximum_key]
                ten_highest_keys.append(maximum_key)
                ten_highest_weights.append(maximum_weight)
                del self.weights[stance][maximum_key]
            print("The 10 highest weighted features for " + stance + " with their corresponding weights are as follows:")
            for w in range(0,10):
                print("Feature: " + ten_highest_keys[w] + ", Weight: " + str(ten_highest_weights[w]))
        for stance in self.CLASSES:
            ten_lowest_keys=[]
            ten_lowest_weights=[]
            minimum_weight=0
            minimum_key=""
            for x in range(0,10):
                minimum_key=min(self.weights[stance], key=self.weights[stance].get)
                minimum_weight=self.weights[stance][minimum_key]
                ten_lowest_keys.append(minimum_key)
                ten_lowest_weights.append(minimum_weight)
                del self.weights[stance][minimum_key]
            print("The 10 lowest weighted features for " + stance + " with their corresponding weights are as follows:")
            for w in range(0,10):
                print("Feature: " + ten_lowest_keys[w] + ", Weight: " + str(ten_lowest_weights[w]))
    
    def test_eval(self, test_docs, test_labels):
        pred_labels = [self.predict(d) for d in test_docs]
        ev = Eval(test_labels, pred_labels)
        return ev.accuracy()
    
# END Class Perceptron
