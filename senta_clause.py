#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:02:25 2017

@author: austinlee
"""
import sys, os, nltk, random
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download()

from sent_perceptron import Perceptron
from baseline_emotion import Classifier
from apis import Rosette, Watson, Aylien, Indico


non_feature_data = []
tweet_data = []
emotion_data = []
emotion_dictionary = {}

def load_docs(tweets):
    # Create parallel lists of documents and labels
    targets, docs, labels = [], [], []
    for tweet in tweets:
        #targets
        targets.append(tweet[1])
        # Append tweet to rawdocs
        docs.append(tweet[2])
        # Append stance to labels
        labels.append(tweet[3])

    return targets, docs, labels

def load_featurized_docs(datasplit):
    targets, rawdocs, labels = load_docs(datasplit)

    # Perceptron data:
    train_docs=rawdocs[0:2531]
    test_docs=rawdocs[2532:]
    train_labels=labels[0:2531]
    test_labels=labels[2532:]

    # api data:
    targets= targets[400:800]
    non_feature_data=rawdocs[400:800]
    rose_labels=labels[400:800]

    assert len(rawdocs)==len(labels)>0,datasplit
    train_featdocs = []
    test_featdocs = []
    for d in train_docs:
        train_featdocs.append(extract_feats(d))
    for e in test_docs:
        test_featdocs.append(extract_feats(e))
    return train_featdocs, train_labels, test_featdocs, test_labels, targets, non_feature_data, rose_labels

def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()

    # Creates an array of words representing each tweet
    doc_split=doc.split()

    # Bias feature
    ff['Bias']=1

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

    """

    # Emotion feature extraction
    # (if a word has an association with an emotion, the boolean for that emotion becomes 1 for the entire tweet)
    for word in doc_split:
        if word in emotion_dictionary:
            for emotion in word:
                ff[emotion]=1
        """
        else:
            similarities = {}
            sets = []

            similarities['anger'] = 0
            similarities['anticipation'] = 0
            similarities['disgust'] = 0
            similarities['fear'] = 0
            similarities['joy'] = 0
            similarities['sadness'] = 0
            similarities['surprise'] = 0
            similarities['trust']= 0

            if wn.synsets(word):
                word_set = wn.synsets(word)[0]
                sets.append(wn.synsets("anger")[0])
                sets.append(wn.synsets("anticipation")[0])
                sets.append(wn.synsets("disgust")[0])
                sets.append(wn.synsets("fear")[0])
                sets.append(wn.synsets("joy")[0])
                sets.append(wn.synsets("sadness")[0])
                sets.append(wn.synsets("surprise")[0])
                sets.append(wn.synsets("trust")[0])

                if not word_set:
                    break
                else:
                    index = 0
                    for _set in sets:
                        if index == 0:
                            s = word_set.wup_similarity(_set)
                            if s != None:
                                similarities['anger'] = s
                        if index == 1:
                            s = word_set.wup_similarity(_set)
                            if s != None:
                                similarities['anticipation'] = s
                        if index == 2:
                            s = word_set.wup_similarity(_set)
                            if s != None:
                                similarities['disgust'] = s
                        if index == 3:
                            s = word_set.wup_similarity(_set)
                            if s != None:
                                similarities['fear'] = s
                        if index == 4:
                            s = word_set.wup_similarity(_set)
                            if s != None:
                                similarities['joy'] = s
                        if index == 0:
                            s = word_set.wup_similarity(_set)
                            if s != None:
                                similarities['sadness'] = s
                        if index == 0:
                            s = word_set.wup_similarity(_set)
                            if s != None:
                                similarities['surprise'] = s
                        if index == 0:
                            s = word_set.wup_similarity(_set)
                            if s != None:
                                similarities['trust'] = s
                        index+=1
                    emo = min(similarities, key=similarities.get)
                    ff[emo]=1
        """

    return ff

def emo_dict(emotion_data):
    for entry in emotion_data:
        # Word does not exist in dictionary, possibly multiple emotions associated with it
        if entry[2]=='1' and entry[0] not in emotion_dictionary:
            emotions=[]
            emotions.append(entry[1])
            emotion_dictionary[entry[0]]=emotions
        # Word exists in dictionary, add relevant emotion to list corresponding to word
        elif entry[2]=='1' and entry[0] in emotion_dictionary:
            emotion_dictionary[entry[0]].append(entry[1])

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
    random.shuffle(tweet_data)


if __name__ == "__main__":


    load_data()
    train_docs, train_labels, test_docs, test_labels, targets, nf_data, rose_labels = load_featurized_docs(tweet_data)


    # Perceptron Model - to use this run: python senta_clause.py #number_of_iterations. i.e python python senta_clause.py 30
    args = sys.argv[1:]
    niters = int(args[0])
    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=test_docs, dev_labels=test_labels)
    acc = ptron.test_eval(test_docs, test_labels)

    # Rosette Sentiment Analysis (API)
    # _rose = Rosette(nf_data, rose_labels)
    # acc = _rose.test_eval()

    #Watson api
    # _watson =  Watson(targets, nf_data, rose_labels)
    # acc = _watson.test_eval()

    #Aylien
    # _ay = Aylien(nf_data, rose_labels)
    # acc = _ay.test_eval()

    #indico
    # _indico = Indico(nf_data, rose_labels)
    # acc = _indico.test_eval()
