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
from rose import Rosette

non_feature_data = []
tweet_data = []
emotion_data = []
emotion_dictionary = {}

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
    
    # Perceptron data:
    train_docs=rawdocs[0:2531]
    test_docs=rawdocs[2532:]
    train_labels=labels[0:2531]
    test_labels=labels[2532:]
    
    # Rosette data:
    non_feature_data=rawdocs[0:5]
    rose_labels=labels[0:5]
    
    assert len(rawdocs)==len(labels)>0,datasplit
    train_featdocs = []
    test_featdocs = []
    for d in train_docs:
        train_featdocs.append(extract_feats(d))
    for e in test_docs:
        test_featdocs.append(extract_feats(e))
    return train_featdocs, train_labels, test_featdocs, test_labels, non_feature_data, rose_labels

def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()
    
    # Creates an array of words representing each tweet
    doc_split=doc.split()
    
    """
    # Unigram feature extraction
    for word in doc_split:
        ff[word]=1
    """
        
    # Bigram feature extraction
    for y in range(1, len(doc_split)):
        current_word = doc_split[y]
        prev_word = doc_split[y-1]
        bigram = prev_word+" "+current_word
        ff[bigram]=1

    # Emotion feature extraction
    # (if a word has an association with an emotion, the boolean for that emotion becomes 1 for the entire tweet)
    for word in doc_split:
        if word in emotion_dictionary:
            ff[emotion_dictionary[word]]=1
    
    return ff

def emo_dict(emotion_data):
    for entry in emotion_data:
        if entry[2]=='1':
            emotion_dictionary[entry[0]]=entry[1]

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
        
    emo_dict(emotion_data)
        

if __name__ == "__main__":
    """
    args = sys.argv[1:]
    niters = int(args[0])
    """
    load_data()
    train_docs, train_labels, test_docs, test_labels, nf_data, rose_labels = load_featurized_docs(tweet_data)
    """
    # Perceptron Model
    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=test_docs, dev_labels=test_labels)
    acc = ptron.test_eval(test_docs, test_labels)
    """
    
    # Rosette Sentiment Analysis (API)
    _rose = Rosette(nf_data, rose_labels)
    acc = _rose.test_eval()
