#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:22:14 2017

@author: austinlee
"""

import json
import tempfile

from rosette.api import API, DocumentParameters, RosetteException
from evaluation import Eval

from watson_developer_cloud import NaturalLanguageUnderstandingV1 as NaturalLanguageUnderstanding
from watson_developer_cloud.natural_language_understanding_v1 import Features, SentimentOptions

from aylienapiclient import textapi

import indicoio

# for watson
natural_language_understanding = NaturalLanguageUnderstanding(
  username="7be10325-9229-4525-a2ee-1f41b7505359",
  password="vDq7SnkScMlM",
  version="2017-02-27")

# for aylien
client = textapi.Client("12ffa925", "5d3e76004b21a0520fe4cafa1d6a729f")

# for Indico
indicoio.config.api_key = '5e121dc4ae2061b353da4b6f9dcee2e0'

result_array=[]
pred_array=[]

class Rosette(object):
    def __init__(self, nf_data, labels):
        self.data=nf_data
        self.test_labels=labels
        self.do_Werk(self.data)
        global result_array
        pred_array = []

    def run(self, key, alt_url, data):
        # Create default file to read from
        temp_file = tempfile.NamedTemporaryFile(suffix=".html")
        sentiment_file_data = "<html><head><title>None</title></head><body><p>" + data + "/p></body></html>"
        message = sentiment_file_data
        temp_file.write(message if isinstance(message, bytes) else message.encode())
        temp_file.seek(0)

        # Create an API instance
        api = API(user_key=key, service_url=alt_url)

        params = DocumentParameters()
        params["language"] = "eng"

        # Use an HTML file to load data instead of a string
        params.load_document_file(temp_file.name)
        try:
            result = api.sentiment(params)
        except RosetteException as exception:
            print(exception)
        finally:
            # Clean up the file
            temp_file.close()

        return result

    def do_Werk(self, data):
        with open('rosette_data.txt', 'w') as outfile:
            for entry in data:
                RESULT = self.run("1c929b776a59d4156ee7d98bbcdcc7f0", 'https://api.rosette.com/rest/v1/', entry)
                result_array.append(RESULT)
                print(json.dump(RESULT, outfile, indent=2))

    def predict(self):
        for _result in result_array:
            if _result["document"]["label"]=='pos':
                pred_array.append("FAVOR")
            elif _result["document"]["label"]=='neg':
                pred_array.append("AGAINST")
            elif _result["document"]["label"]=='neu':
                pred_array.append("NONE")

    def test_eval(self):
        self.predict()
        ev = Eval(self.test_labels, pred_array)
        return ev.accuracy()

class Watson(Rosette):
    def __init__(self, targets, nf_data, labels):
        self.targets=targets
        self.data=nf_data
        self.test_labels=labels
        self.do_Werk(self.targets, self.data)
        global result_array
        pred_array = []

    def run(self, target, data):
        response = natural_language_understanding.analyze(
            text=target + " " + data,
            features=Features(sentiment=SentimentOptions(targets=[target])))["sentiment"]["targets"][0]["label"]
        return response

    def do_Werk(self, targets, data):
        with open('watson_data.txt', 'w') as outfile:
            for index, entry in enumerate(data):
                RESULT = self.run(targets[index], entry)
                result_array.append(RESULT)
                print(json.dump(RESULT, outfile, indent=2))

    def predict(self):
        for _result in result_array:
            if _result == 'positive':
                pred_array.append("FAVOR")
            elif _result == 'negative':
                pred_array.append("AGAINST")
            elif _result =='neutral':
                pred_array.append("NONE")

class Aylien(Watson):
    def __init__(self, nf_data, labels):
        self.data=nf_data
        self.test_labels=labels
        self.do_Werk(self.data)
        global result_array
        pred_array = []

    def run(self, data):
        sentiment = client.Sentiment({'text': data})["polarity"]
        print(sentiment)
        return sentiment

    def do_Werk(self, data):
        with open('aylien_data.txt', 'w') as outfile:
            for entry in data:
                RESULT = self.run(entry)
                result_array.append(RESULT)
                print(json.dump(RESULT, outfile, indent=2))


class Indico(Rosette):
    def run(self, data):
        sentiment = indicoio.sentiment("data")
        return sentiment

    def do_Werk(self, data):
        with open('Indico_data.txt', 'w') as outfile:
            for entry in data:
                RESULT = self.run(entry)
                result_array.append(RESULT)
                json.dump(RESULT, outfile, indent=2)


    def predict(self):
        self.test_labels.append("FAVOR")
        self.test_labels.append("AGAINST")
        self.test_labels.append("NONE")
        pred_array.append("FAVOR")
        pred_array.append("AGAINST")
        pred_array.append("NONE")
        for _result in result_array:
            if _result > .5:
                pred_array.append("FAVOR")
            elif _result < .5:
                pred_array.append("AGAINST")
            elif _result == .5:
                pred_array.append("NONE")
