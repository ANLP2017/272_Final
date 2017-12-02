#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:02:25 2017

@author: austinlee
"""
import sys, os, nltk
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download()

from sent_perceptron import Perceptron
from baseline_emotion import Classifier

tweet_data = []
emotion_data = []
niters=30

def load_docs(tweets):
    # Create parallel lists of documents and labels
    docs, labels = [], []
    for tweet in tweets:
        # Append tweet to rawdocs
        docs.append(tweet[2])
        # Append stance to labels
        labels.append(tweet[3])
        
    return docs, labels

def load_featurized_docs(datasplit):
    rawdocs, labels = load_docs(datasplit)
    train_docs=rawdocs[0:2531]
    test_docs=rawdocs[2532:]
    train_labels=labels[0:2531]
    test_labels=labels[2532:]
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in train_docs:
        featdocs.append(extract_feats(d))
    return featdocs, train_labels, test_docs, test_labels

def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()

    # very basic tokenization
    for word in doc.split(" "):
        ff[word]=1

    return ff


def load_data():
    # Open and read in training data into tweet_data[[]] array of arrays
    # FORMAT: tweet_data[x][0]=ID, tweet_data[x][1]=TOPIC, tweet_data[x][2]=TWEET, tweet_data[x][3]=STANCE
    with open("training_data.txt",'r', encoding="utf-8") as inFile:
        for line in inFile:
            line=line.rstrip('\n')
            data=line.split("\t")
            tweet_data.append(data)
        del tweet_data[0] # First line does not contain data

    # Open and read in emotion data into emotion_Data[[]] array of arrays
    # FORMAT: emotion_data[x][0]=WORD, emotion_data[x][1]=EMOTION, emotion_data[x][2]=ASSOCIATION     
    with open("emotion_data.txt", 'r') as nextInFile:
        for line in nextInFile:
            line=line.rstrip('\n')
            data=line.split('\t')
            emotion_data.append(data)
        del emotion_data[0] # First line does not contain data
        

if __name__ == "__main__":
    load_data()
    train_docs, train_labels, test_docs, test_labels = load_featurized_docs(tweet_data)
    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=test_docs, dev_labels=test_labels)
    #acc = ptron.test_eval(test_docs, test_labels)
