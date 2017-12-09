from sklearn import svm
from sklearn.cluster import KMeans
import json
import re
from nltk.stem import WordNetLemmatizer
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

STANCES = ['AGAINST', 'FAVOR', 'NONE']
TOPICS = ['Atheism', 'Climate Change is a Real Concern', 'Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion',]

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
			i = 0
			for tweet in tweets:
				cleaned_tweet = []

				for word in tweet[2].split():
					cleaned_tweet.append(self.lemmatizer.lemmatize(re.sub(self.punc, '', word)).lower())
					pass

				for index, word in enumerate(cleaned_tweet):
					if word == '':
						del cleaned_tweet[index]

				cleaned_tweets.append(cleaned_tweet)

		with open('train_docs.json', 'r') as f:
			train_data = json.load(f)
			train_data = [(list(doc)) for doc in train_data]

			for doc in train_data:
				for index, word in enumerate(doc):
					doc[index] = re.sub(self.punc, '', word)
					doc[index] = self.lemmatizer.lemmatize(word)
	
		with open('train_labels.json', 'r') as f:
			train_labels = json.load(f)

		with open('test_docs.json', 'r') as f:
			test_data = json.load(f)
			test_data = [(list(doc)) for doc in test_data]

			for doc in test_data:
				for index, word in enumerate(doc):
					doc[index] = re.sub(self.punc, '', word)
			
		with open('test_labels.json', 'r') as f:
			test_labels = json.load(f)

		train_data = cleaned_tweets[:2531]
		test_data = cleaned_tweets[2532:]

		self.original_train = train_data
		self.original_test = test_data
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

class SplitProcessing:
	def __init__(self):
		self.lemmatizer = WordNetLemmatizer()
		self.punc = r'[^A-za-z]'
		self.train_docs, self.train_labels, self.test_docs, self.test_labels = self.load_data()
		self.setup_sentiment_lexicons(self.train_docs)

	def load_data(self):
		with open('tweet_bank.json', 'r') as f:
			tweets = json.load(f)

			label_indices = {}
			grouped_tweets = {t: {} for t in TOPICS}
			for index, tweet in enumerate(tweets):
				cleaned_tweet = []

				for word in tweet[2].split():
					cleaned_tweet.append(self.lemmatizer.lemmatize(re.sub(self.punc, '', word)).lower())

				for ndx, word in enumerate(cleaned_tweet):
					if word == '':
						del cleaned_tweet[ndx]

				label_indices[index] = tweet[-1]
				grouped_tweets[tweet[1]][index] = cleaned_tweet

		train_docs = {t: {} for t in TOPICS}
		train_labels = {t: {} for t in TOPICS}
		test_docs = {t: {} for t in TOPICS}
		test_labels = {t: {} for t in TOPICS}

		for topic in grouped_tweets:
			for tweet in grouped_tweets[topic]:
				chance = random.randint(0, 100)
				if chance <= 85:
					train_docs[topic][tweet] = grouped_tweets[topic][tweet]
					train_labels[topic][tweet] = label_indices[tweet]
				else:
					test_docs[topic][tweet] = grouped_tweets[topic][tweet]
					test_labels[topic][tweet] = label_indices[tweet]

		return train_docs, train_labels, test_docs, test_labels

	def setup_sentiment_lexicons(self, docs):
		with open('keyed_emotion_lexicon.json', 'r') as f:
			lexicon = json.load(f)

		count = 0
		index_word_map = {}

		vocab = set()
		for topic in docs:
			for doc in docs[topic]:
				for word in docs[topic][doc]:
					vocab.add(word.lower())

		self.at_map = {}
		self.cc_map = {}
		self.fm_map = {}
		self.hc_map = {}
		self.ab_map = {}

		for topic in docs:
			for word in lexicon:
				if sum([int(val) for val in lexicon[word].values()]) != 0 and word in vocab:
					if topic == 'Atheism':
						self.at_map[word] = count
					elif topic == 'Climate Change is a Real Concern':
						self.cc_map[word] = count
					elif topic == 'Feminist Movement':
						self.fm_map[word] = count
					elif topic == 'Hillary Clinton':
						self.hc_map[word] = count
					else:
						self.ab_map[word] = count
					count += 1
			count = 0

	def get_sentiment_features(self, topic, doc, binary=True):
		if topic == 'Atheism':
			features = [0 for i in range(len(self.at_map))]
		elif topic == 'Climate Change is a Real Concern':
			features = [0 for i in range(len(self.cc_map))]
		elif topic == 'Feminist Movement':
			features = [0 for i in range(len(self.fm_map))]
		elif topic == 'Hillary Clinton':
			features = [0 for i in range(len(self.hc_map))]
		else:
			features = [0 for i in range(len(self.ab_map))]

		for word in doc:
			if topic == 'Atheism':
				if word.lower() in self.at_map.keys():
					if binary:
						features[self.at_map[word.lower()]] = 1
					else:
						features[self.at_map[word.lower()]] += 1
			elif topic == 'Climate Change is a Real Concern':
				if word.lower() in self.cc_map.keys():
					if binary:
						features[self.cc_map[word.lower()]] = 1
					else:
						features[self.cc_map[word.lower()]] += 1
			elif topic == 'Feminist Movement':
				if word.lower() in self.fm_map.keys():
					if binary:
						features[self.fm_map[word.lower()]] = 1
					else:
						features[self.fm_map[word.lower()]] += 1
			elif topic == 'Hillary Clinton':
				if word.lower() in self.hc_map.keys():
					if binary:
						features[self.hc_map[word.lower()]] = 1
					else:
						features[self.hc_map[word.lower()]] += 1
			else:
				if word.lower() in self.ab_map.keys():
					if binary:
						features[self.ab_map[word.lower()]] = 1
					else:
						features[self.ab_map[word.lower()]] += 1
		
		return features

	@property
	def sentiment_feature_docs(self):
		feat_train_docs = {t: [] for t in TOPICS}
		feat_train_labels = {t: [] for t in TOPICS}

		for topic in self.train_docs:
			for doc in sorted(self.train_docs[topic]):
				feat_train_docs[topic].append(self.get_sentiment_features(topic, self.train_docs[topic][doc], binary=True))

		for topic in self.train_labels:
			for doc in sorted(self.train_labels[topic]):
				feat_train_labels[topic].append(self.train_labels[topic][doc])

		feat_test_docs = {t: [] for t in TOPICS}
		feat_test_labels = {t: [] for t in TOPICS}
		for topic in self.test_docs:
			for doc in sorted(self.test_docs[topic]):
				feat_test_docs[topic].append(self.get_sentiment_features(topic, self.test_docs[topic][doc], binary=True))

		for topic in self.test_labels:
			for doc in sorted(self.test_labels[topic]):
				feat_test_labels[topic].append(self.test_labels[topic][doc])

		return feat_train_docs, feat_train_labels, feat_test_docs, feat_test_labels

class SentaClauseClustering:
	def __init__(self, train_docs=None, train_labels=None, test_docs=None, test_labels=None):
		self.train_docs = train_docs
		self.train_labels = train_labels
		self.test_docs = test_docs
		self.test_labels = test_labels

		self.clustering_clf = KMeans(n_clusters=2)

	def classifier(self):
		self.clustering_clf.fit(self.train_docs)

	def make_predictions(self):
		self.predictions = self.clustering_clf.predict(self.test_docs)

class SentaClauseSvm:
	def __init__(self, train_docs=None, train_labels=None, test_docs=None, test_labels=None):
		self.train_docs = train_docs
		self.train_labels = train_labels
		self.test_docs = test_docs
		self.test_labels = test_labels

		self.svm_clf = svm.SVC(decision_function_shape='ovo')

	def classifier(self):
		self.svm_clf.fit(self.train_docs, self.train_labels)

	def make_predictions(self):
		self.predictions = self.svm_clf.predict(self.test_docs)

class SentaClauseNB:
	def __init__(self, train_docs=None, train_labels=None, test_docs=None, test_labels=None):
		self.train_docs = train_docs
		self.train_labels = train_labels
		self.test_docs = test_docs
		self.test_labels = test_labels

		self.nb_clf = MultinomialNB()

	def classifier(self):
		self.nb_clf.fit(self.train_docs, self.train_labels)

	def make_predictions(self):
		self.predictions = self.nb_clf.predict(self.test_docs)


random.seed(0)
processing = Preprocessing()
all_docs = processing.sentiment_feature_docs

single_baseline = {s: 0 for s in STANCES}
for label in processing.train_labels:
	single_baseline[label] += 1

for label in single_baseline:
	single_baseline[label] /= len(processing.train_labels)

print('TRAINING DATA CLASS PROPORTIONS:\n{}'.format(single_baseline))

test = SentaClauseSvm(
	all_docs[0], 
	processing.train_labels, 
	all_docs[1], 
	processing.test_labels)
test.classifier()
test.make_predictions()

print('TOPIC AGNOSTIC CLASSIFIERS\nCLUSTERING\nPREDICTIONS:')
kmeans = SentaClauseClustering(
	all_docs[0], 
	processing.train_labels, 
	all_docs[1], 
	processing.test_labels)
kmeans.classifier()
kmeans.make_predictions()
print('{}\nSCORE: {}'.format(kmeans.predictions, metrics.silhouette_score(kmeans.test_docs, kmeans.test_labels, metric='euclidean')))

nb = SentaClauseNB(
	all_docs[0], 
	processing.train_labels, 
	all_docs[1], 
	processing.test_labels)
nb.classifier()
nb.make_predictions()

print('NAIVE BAYES\nPREDICTIONS:\n{}'.format(nb.predictions))
good = 0
total = 0
for index, p in enumerate(nb.predictions):
	if p == processing.test_labels[index]:
		good += 1
	total += 1
print('ACCURACY: {}\n'.format(good/total))

sp = SplitProcessing()
print('TOPIC SPECIFIC CLASSIFIERS\n NAIVE BAYES')
fd = sp.sentiment_feature_docs
for topic in TOPICS:
	baseline = {s: 0 for s in STANCES}
	for label in fd[1][topic]:
		baseline[label] += 1
	for label in baseline:
		baseline[label] /= len(fd[1][topic])

	good = 0
	total = 0

	snb = SentaClauseNB(fd[0][topic], fd[1][topic], fd[2][topic], fd[3][topic])
	snb.classifier()
	snb.make_predictions()
	print('{}\n PREDICTIONS:\n{}'.format(topic, snb.predictions))

	for index, p in enumerate(snb.predictions):
		if p == fd[3][topic][index]:
			good += 1
		total += 1

	print('ACCURACY: {}\nTRAINING CLASS PROPORTIONS:\n{}\n'.format(good/total, baseline))

print('TOPIC SPECIFIC CLASSIFIERS\n CLUSTERING')
for topic in TOPICS:
	# baseline = {s for s in STANCES}
	# for label in fd[1][topic]:
	# 	baseline[label] += 1
	# for label in baseline:
	# 	baseline[label] /= len(fd[1][topic])

	skm = SentaClauseClustering(fd[0][topic], fd[1][topic], fd[2][topic], fd[3][topic])
	skm.classifier()
	skm.make_predictions()
	print('{}\n PREDICTIONS:\n{}\n'.format(topic, skm.predictions))
	print(fd[3][topic])
	# print('SCORE: {}'.format(skm.clustering_clf.score(fd[2][topic])))
	print('{}'.format(metrics.silhouette_score(fd[2][topic], fd[3][topic], metric='euclidean')))
