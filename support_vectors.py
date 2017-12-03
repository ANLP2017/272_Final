from sklearn import svm
from sklearn.cluster import KMeans
import numpy as np
import json
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
import random

# input two arrays: an array X of size [n_samples, n_features] holding the training samples, 
# and an array y of class labels (strings or integers), size [n_samples]

class Preprocessing:
	def __init__(self):
		self.lemmatizer = WordNetLemmatizer()
		self.punc = r'[^A-za-z]'
		self.train_docs, self.train_labels, self.test_docs, self.test_labels = self.load_data()
		self.setup_sentiment_lexicon(self.train_docs)

	def load_data(self):
		
		with open('tweet_bank.json', 'r') as f:
			tweets = json.load(f)

			cleaned_tweets = []
			for tweet in tweets:
				cleaned_tweet = []

				for word in tweet[2].split():
					cleaned_tweet.append(self.lemmatizer.lemmatize(re.sub(self.punc, '', word)).lower())
					pass

				for index, word in enumerate(cleaned_tweet):
					if word == '':
						del cleaned_tweet[index]

				cleaned_tweets.append(cleaned_tweet)

			# random.shuffle(cleaned_tweets)

		with open('train_docs.json', 'r') as f:
			train_data = json.load(f)
			train_data = [(list(doc)) for doc in train_data]

			for doc in train_data:
				for index, word in enumerate(doc):
					doc[index] = re.sub(self.punc, '', word)
					doc[index] = self.lemmatizer.lemmatize(word)

			# import pdb; pdb.set_trace()
			# pass
	
		with open('train_labels.json', 'r') as f:
			train_labels = json.load(f)
			# train_labels = np.array(train_labels)

		with open('test_docs.json', 'r') as f:
			test_data = json.load(f)
			test_data = [(list(doc)) for doc in test_data]

			for doc in test_data:
				for index, word in enumerate(doc):
					doc[index] = re.sub(self.punc, '', word)
			
		with open('test_labels.json', 'r') as f:
			test_labels = json.load(f)
			# test_labels = np.array(test_labels)

		# import pdb; pdb.set_trace()
		# pass

		train_data = cleaned_tweets[:2531]
		test_data = cleaned_tweets[2532:]

		return train_data, train_labels, test_data, test_labels

	def setup_sentiment_lexicon(self, docs):
		with open('keyed_emotion_lexicon.json', 'r') as f:
			lexicon = json.load(f)

		count = 0
		index_word_map = {}

		vocab = set()
		for doc in docs:
			for word in doc:
				vocab.add(word.lower())

		self.lexicon = []
		self.index_word_map = {}
		for word in lexicon:
			if sum([int(val) for val in lexicon[word].values()]) != 0 and word in vocab:
				# index_word_map[word] = count
				# import pdb; pdb.set_trace()
				# pass
				self.lexicon.append(word)
				self.index_word_map[word] = count
				count += 1

	@property
	def sentiment_feature_docs(self):
		feat_train_docs = []
		for doc in self.train_docs:
			feat_train_docs.append(self.get_sentiment_features(doc, binary=True))

		feat_test_docs = []
		for doc in self.test_docs:
			feat_test_docs.append(self.get_sentiment_features(doc, binary=True))

		return feat_train_docs, feat_test_docs

	def get_sentiment_features(self, doc, binary=False):
		# doc is iterable of words

		# index_word_map = {entry[1]: entry[0] for entry in enumerate(lexicon) if sum([int(val) for val in lexicon[entry[1]].values()]) != 0}
		features = [0 for i in range(len(self.index_word_map))]

		for word in doc:
			if word.lower() in self.index_word_map.keys():
				if binary:
					features[self.index_word_map[word.lower()]] = 1
				else:
					features[self.index_word_map[word.lower()]] += 1
		
		return features

class SentaClauseClustering:
	def __init__(self, train_docs=None, train_labels=None, test_docs=None, test_labels=None):
		self.train_docs = train_docs
		self.train_labels = train_labels
		self.test_docs = test_docs
		self.test_labels = test_labels

		self.clustering_clf = KMeans(n_clusters=3)

	def classifier(self):
		self.clustering_clf.fit(self.train_docs)

	def make_predictions(self):
		self.predictions = self.clustering_clf.predict(self.test_docs)

class SentaClauseSvm:
	def __init__(self, train_docs=None, train_labels=None, test_docs=None, test_labels=None):
		# processing = Preprocessing()
		# self.train_docs, self.test_docs = processing.sentiment_feature_docs
		# self.train_labels = processing.train_labels
		# self.test_labels = processing.test_labels
		self.train_docs = train_docs
		self.train_labels = train_labels
		self.test_docs = test_docs
		self.test_labels = test_labels

		self.svm_clf = svm.SVC(decision_function_shape='ovo')

	def classifier(self):
		# import pdb; pdb.set_trace()
		# pass
		self.svm_clf.fit(self.train_docs, self.train_labels)

	def make_predictions(self):
		self.predictions = self.svm_clf.predict(self.test_docs)

processing = Preprocessing()
test = SentaClauseSvm(
	processing.sentiment_feature_docs[0], 
	processing.train_labels, 
	processing.sentiment_feature_docs[1], 
	processing.test_labels)
test.classifier()
test.make_predictions()

# import pdb; pdb.set_trace()
# pass
kmeans = SentaClauseClustering(
	processing.sentiment_feature_docs[0], 
	processing.train_labels, 
	processing.sentiment_feature_docs[1], 
	processing.test_labels)
kmeans.classifier()
kmeans.make_predictions()
# import pdb; pdb.set_trace()
# pass

# plt.figure(1)
# plt.clf()

# for doc in kmeans.train_docs:
# 	plt.plot(doc, [0] * 1460)

# plt.show()

