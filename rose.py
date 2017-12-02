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

result_array=[]
pred_array=[]

class Rosette():
    def __init__(self, nf_data, labels):
        self.data=nf_data
        self.test_labels=labels
        self.do_Werk(self.data)
    
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
        