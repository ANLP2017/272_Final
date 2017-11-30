#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:41:10 2017

@author: austinlee
"""
class Classifier:
    def _init_(self, train_docs, train_labels, emotions):
        self.classify(train_docs, train_labels, emotions)
        
    def classify(self, train_docs, train_labels, emotions):
        """
        IDEAS:
            
        BASELINE EMOTION CLASSIFIER:
        1. Analyse each tweet's raw word data, searching each word in the emotions 
           data stucture to find the word's respective emotion (one of 8 emotions: 
           anger, joy,anticipation, disgust, fear, sadness, surprise, and trust). 
           Disregard any data with associations to negative and positive. 
           
        2. Evaluate which emotions have associations (value of 1) for the given word
        
        3. sum the totals for the 2 groups of four emotions (positive: joy, anticipation, 
           surprise, trust; negative: anger, disgust, fear, sadness)
        
        4. See which of the two groups has a higher sum, and decide whether or
           not the tweet is AGAINST(negative sum is greater), FAVOR(positive sum is greater),
           or NONE(sums are equal)
        
        BASELINE EMOTION CLASSIFIER W/ WORDNET:
        1. Analyse each tweet's raw word data. Evaluate the wordnet distance of each
           word to each of the 8 emotions listed above. 
           
        2. Find which emotion has the closest wordnet distance to each word.
        
        3. Sum all of the emotions.
        
        4. The emotion with the highest sum dictates positive or negative, based off of
           the classification of the emotion (see above)
        """