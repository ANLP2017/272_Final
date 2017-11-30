#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:02:25 2017

@author: austinlee
"""
import sys, os, glob, nltk, argparse, re
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download()

from sent_perceptron import Perceptron
from baseline_emotion import Classifier

tweet_data = []
emotion_data = []

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
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in rawdocs:
        featdocs.append(extract_feats(d))
    return featdocs, labels

def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()
    

    return ff


def load_data():
    # Open and read in training data into tweet_data[[]] array of arrays
    # FORMAT: tweet_data[x][0]=ID, tweet_data[x][1]=TOPIC, tweet_data[x][2]=TWEET, tweet_data[x][3]=STANCE
    with open("training_data.txt",'r', encoding="utf-8") as inFile:
        print("File opened successfully")
        for line in inFile:
            line=line.rstrip('\n')
            data=line.split("\t")
            tweet_data.append(data)
        del tweet_data[0] # First line does not contain data

    # Open and read in emotion data into emotion_Data[[]] array of arrays
    # FORMAT: emotion_data[x][0]=WORD, emotion_data[x][1]=EMOTION, emotion_data[x][2]=ASSOCIATION     
    with open("emotion_data.txt", 'r') as nextInFile:
        print("File opened successfully")
        for line in nextInFile:
            line=line.rstrip('\n')
            data=line.split('\t')
            emotion_data.append(data)
        del emotion_data[0] # First line does not contain data
        

if __name__ == "__main__":
    load_data()
    load_featurized_docs(tweet_data)
