import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1 as NaturalLanguageUnderstanding
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
from evaluation import Eval

natural_language_understanding = NaturalLanguageUnderstanding(
  username="7be10325-9229-4525-a2ee-1f41b7505359",
  password="vDq7SnkScMlM",
  version="2017-02-27")

result_array=[]
pred_array=[]

class Watson():
    def __init__(self, nf_data, labels):
        self.data=nf_data
        self.test_labels=labels
        self.do_Werk(self.data)

    def run(self, data):
        response = natural_language_understanding.analyze(
          text=data,
          features=Features(entities=EntitiesOptions(
                                  emotion=True, sentiment=True,limit=2),
                           keywords=KeywordsOptions(
                                  emotion=True, sentiment=True,limit=2
                                            ))).get("keywords")[0]["sentiment"]["label"]
        return response

    def do_Werk(self, data):
        with open('watson_data.txt', 'w') as outfile:
            for entry in data:
                RESULT = self.run(entry)
                result_array.append(RESULT)
                print(json.dump(RESULT, outfile, indent=2))

    def predict(self):
        for _result in result_array:
            if _result =='positive':
                pred_array.append("FAVOR")
            elif _result =='negative':
                pred_array.append("AGAINST")
            elif _result =='neutral':
                pred_array.append("NONE")

    def test_eval(self):
        self.predict()
        ev = Eval(self.test_labels, pred_array)
        return ev.accuracy()
