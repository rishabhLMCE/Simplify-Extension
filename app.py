label = [[1, 'First Party Collection/Use'], 
            [2, 'Third Party Sharing/Collection'], 
            [3, 'User Choice/Control'], 
            [4, 'User Access, Edit and Deletion'], 
            [5, 'Data Retention'],
            [6, 'Data Security'],
            [7, 'Policy Change'], 
            [8, 'Do Not Track'],
            [9, 'International and Specific Audiences'],
            [10, 'Introductory/Generic'],
            [11, 'Privacy contact information'],
            [12, 'Privacy contact information']]

import os
import json
import csv
import pandas as pd
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report


def readPolicyFile(fileLocation):
  policySegments = []
  for filename in os.listdir(fileLocation):
    try:
      absFilename = "{}/{}".format(fileLocation,filename)
      #with open(absFilename) as csv_file:
      #print absFilename
      categoryId = 0
      #print(absFilename)
      df2 = pd.read_csv(absFilename)
        #csv_reader = csv.reader(csv_file, delimiter=',')
        #csv_reader = unicode_csv_reader(open(csv_file))
      for i  in range(df2.shape[0]):
          if df2.iloc[i][5] == "First Party Collection/Use":
              categoryId = 1
          elif df2.iloc[i][5] == "Third Party Sharing/Collection":
              categoryId = 2
          elif  df2.iloc[i][5] == "User Choice/Control":
              categoryId = 3
          elif  df2.iloc[i][5] == "User Access, Edit and Deletion":
              categoryId = 4
          elif  df2.iloc[i][5] == "Data Retention":
              categoryId = 5
          elif  df2.iloc[i][5] == "Data Security":
              categoryId = 6
          elif  df2.iloc[i][5] == "Policy Change":
              categoryId = 7
          elif  df2.iloc[i][5] == "Do Not Track":
              categoryId = 8 
          elif  df2.iloc[i][5] == "International and Specific Audiences":
              categoryId = 9
          elif  df2.iloc[i][5] == "Introductory/Generic":
              categoryId = 10
          elif  df2.iloc[i][5] == "Privacy contact information":
              categoryId = 11
          elif  df2.iloc[i][5] == "Practice not covered":
              categoryId = 12
          else:
              continue
              
          policySegment = ''
          jsonData=json.loads(df2.iloc[i][6])
          for (k, v) in jsonData.items():
              for (k, v) in v.items():
                  if k == 'selectedText':
                      policySegment = ''.join(v)
          
          policySegments.append([policySegment, categoryId, df2.iloc[i][5]])
    except:
        pass
      #print policySegments
      #print policySegments
  df = pd.DataFrame(policySegments, columns = ['text', 'label', 'label_name'])
  return df


def cleanDocs(dataFrame):
  cleanNull = dataFrame[df.text != 'null'].reset_index(drop=True)
  stop = set(stopwords.words('english'))
  exclude = set(string.punctuation) 
  lemma = WordNetLemmatizer()
  clean_docs = []
  bigram_docs = []
  for index, entry in enumerate(cleanNull['text']):
    try:
      stop_free = " ".join([i for i in entry.lower().split() if i not in stop])
      punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
      digit_free = [word for word in punc_free.split() if not word.isdigit() and len(word) > 2]
      normalized = " ".join(lemma.lemmatize(word) for word in digit_free)
      nouns = [word[0] for word in nltk.pos_tag(normalized.split()) if word[1] == 'NN' or word[1] == 'VB']
      cleanNull.loc[index,'text_final'] = str(nouns)
    except:
      pass
#bigram_transformer = phrases.Phrases(clean_docs)

#for doc in bigram_transformer[clean_docs]:
#		bigram_docs.append(doc)
  cleanEmpty = cleanNull[cleanNull.text_final != '[]']
  return cleanEmpty

def loadTestDataset(fileName):
  df = pd.read_csv(fileName)
  df=df.dropna()
  return df

def buildModel(Corpus):
  Train_data, Test_data, Train_label, Test_label = train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)
  #Encoder = preprocessing.LabelEncoder()
  #Train_label = Encoder.fit_transform(Train_label)
  #Test_label = Encoder.fit_transform(Test_label)
  Tfidf_vect = TfidfVectorizer(max_features=50)
  Tfidf_vect.fit(Corpus['text_final'])
  Train_data_Tfidf = Tfidf_vect.transform(Train_data)
  Test_data_Tfidf = Tfidf_vect.transform(Test_data)
  
#     print(Tfidf_vect.vocabulary_)
  
  # fit the training dataset on the NB classifier
#     Naive = MultinomialNB()
#     Naive.fit(Train_data_Tfidf,Train_label)
#     # predict the labels on validation dataset
#     predictions_NB = Naive.predict(Test_data_Tfidf)
#     # Use accuracy_score function to get the accuracy
#     print(classification_report(Test_label, predictions_NB))
#     print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_label)*100)
  
  # Classifier - Algorithm - SVM
  # fit the training dataset on the classifier
  SVM = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
  SVM.fit(Train_data_Tfidf,Train_label)
  # predict the labels on validation dataset
  predictions_SVM = SVM.predict(Test_data_Tfidf)
  #print Test_data_Tfidf
  #print predictions_SVM
  # Use accuracy_score function to get the accuracy
  print(classification_report(Test_label, predictions_SVM))
  print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_label)*100)
  
  
  return SVM

def predictLabel(model, corpus):
  
  Tfidf_vect = TfidfVectorizer(max_features=50)
  Tfidf_vect.fit(corpus['text_final'])
  webPolicy_TFidf = Tfidf_vect.transform(corpus['text_final'])
  webPolicyPrediction = model.predict(webPolicy_TFidf)
  

  return webPolicyPrediction,corpus['text_final'];

def mergeData(corpus, predictedResult):
  labels = [[1, 'First Party Collection/Use'], 
            [2, 'Third Party Sharing/Collection'], 
            [3, 'User Choice/Control'], 
            [4, 'User Access, Edit and Deletion'], 
            [5, 'Data Retention'],
            [6, 'Data Security'],
            [7, 'Policy Change'], 
            [8, 'Do Not Track'],
            [9, 'International and Specific Audiences'],
            [10, 'Introductory/Generic'],
            [11, 'Privacy contact information'],
            [12, 'Privacy contact information']]
  
  dfLabel = pd.DataFrame(labels, columns=['label', 'discription'])
  dfPredictedResult = pd.DataFrame(predictedResult)
  dfContact = pd.concat([corpus, dfPredictedResult], axis=1)
  dfContact.columns = ['topic_number', 'corpus', 'label'] 
  return pd.merge(dfContact, dfLabel, on='label')

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')





# np.random.seed(500)
# fileLocation = '/content/drive/MyDrive/annotations'
# df = readPolicyFile(fileLocation)
# Corpus = cleanDocs(df)
# print (Corpus['label_name'].unique())
# Corpus.to_csv('clean_OOP-115_policy_corpus.csv', index=False)
# # model = buildModel(Corpus)

# Corpus.to_csv('/content/drive/MyDrive/annotations/clean_OOP-115_policy_corpus.csv', index=False)

# model = buildModel(Corpus)

import pickle
filename = 'policy_model.sav'

model = pickle.load(open(filename, 'rb'))

def get_key(val):
  for key, value in mapping.items():
        if val == value:
            return key

  return "key doesn't exist"

def predictPolicyLabel(text):
  data=text
  data = data.split('\n')
  finaldata = []
  for i in data:
    if len(i.split(" "))>4:
      finaldata.append(i)
  # Parse the data, assigning every other row to a different column
  col1 = [str(finaldata[i]) for i in range(0,len(finaldata),2)]
  # Create the data frame
  corpus= pd.DataFrame({'text': col1})
  # corpus=corpus['text'].astype(str)
  corpus=corpus['text'].str.replace('\d+', '')
  corpus=corpus.dropna()
  corpus.to_csv("out.csv",index=False)
  corpus = loadTestDataset("out.csv")
  k,textFinal= predictLabel(model, cleanDocs(corpus))
  l=list(k)
  finalResult = pd.DataFrame()
  finalLabel = [label[i][1] for i in l]
  finalResult["text"] =  corpus["text"]
  finalResult["Label"] = finalLabel
  
  # finalResult = finalResult.to_dict()

  values = finalResult['Label'].value_counts().to_dict()
  mapping = dict(label)

  labelsList = list(values.keys())

  freq = {}
  for i in range(1,13):
    if mapping[i] in list(values.keys()):
      freq[i] = values[mapping[i]]
    else:
      freq[i] = 0

  list_ = []
  for i in range(1, 13):
    if i == 3:
      pass
    elif i == 4:
      pass
    else:
      try:
        list_.append(freq[i])
      except:
        pass

    finalLabel = ""
    val = freq[3] + freq[4]
    if freq[1] > freq[2]:
      if freq [6] > freq[5]:
        if any(y>val for y in list_ ):
          finalLabel ="A"
        elif any(y<=val for y in list_ ):
          finalLabel ="B"
    elif freq[1] <=freq[2]:
      if freq [6] <= freq[5]:
        if any(y>=val for y in list_ ):
          finalLabel ="C"
        elif any(y<=val for y in list_ ):
          finalLabel ="D"
      elif freq[6] > freq[5]:
        if any(y>=val for y in list_ ):
          finalLabel ="D"
        if any(y<=val for y in list_ ):
          finalLabel ="E"

  return finalResult,finalLabel,labelsList




from urllib.request import urlopen
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, url_for, redirect, session #importing libraries
from flask import Flask, request, url_for,Response, render_template_string
import pandas as pd
from flask import Flask, render_template, request


from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import requests
import json
from bson.json_util import dumps
from flask_cors import CORS, cross_origin
from flask import Flask

app = Flask(__name__)
CORS(app)

@app.route("/classify",  methods=['GET',"POST"])
def classify():
    k = request.get_json(force=True)
    url = k["url"]
    try:
      # url = request.args["url"]
      print(url)
      html = urlopen(url).read()
      soup = BeautifulSoup(html, features="html.parser")

      # kill all script and style elements
      for script in soup(["script", "style"]):
          script.extract()    # rip it out

      # get text
      text = soup.get_text()

      # break into lines and remove leading and trailing space on each
      lines = (line.strip() for line in text.splitlines())
      # break multi-headlines into a line each
      chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
      # drop blank lines
      text = '\n'.join(chunk for chunk in chunks if chunk)
      result = {}
      result["text"] = text 
      # print(text)
      # predictPolicyLabel(text).to_csv("result {}.csv".format(url.split("//")[1]))

      result,label,labelList = predictPolicyLabel(text)
      # label = [[1, 'First Party Collection/Use'], 
#               [2, 'Third Party Sharing/Collection'], 
#               [3, 'User Choice/Control'], 
#               [4, 'User Access, Edit and Deletion'], 
#               [5, 'Data Retention'],
#               [6, 'Data Security'],
#               [7, 'Policy Change'], 
#               [8, 'Do Not Track'],
#               [9, 'International and Specific Audiences'],
#               [10, 'Introductory/Generic'],
#               [11, 'Privacy contact information'],
#               [12, 'Privacy contact information']]

      for i in range(len(labelList)):
        if labelList[i] == "First Party Collection/Use":
          labelList[i] = "<h4>{}</h4>".format(labelList[i]) + "<p>{}</p>".format("Information a website collects directly from its users and owns")
        
        if labelList[i] == "Third Party Sharing/Collection":
          labelList[i] = "<h4>{}</h4>".format(labelList[i]) + "<p>{}</p>".format("When a website share their user's information with another firm")
      
        if labelList[i] == "Introductory/Generic":
          labelList[i] = "<h4>{}</h4>".format(labelList[i]) + "<p>{}</p>".format("General introduction about the website")
      
        if labelList[i] == "Do Not Track":
          labelList[i] = "<h4>{}</h4>".format(labelList[i]) + "<p>{}</p>".format("Option of opting out of all the possible tracking")
      
        if labelList[i] == "Data Retention":
          labelList[i] = "<h4>{}</h4>".format(labelList[i]) + "<p>{}</p>".format("Till how long a website will retain its user’s data")

        if labelList[i] == "Policy Change":
          labelList[i] = "<h4>{}</h4>".format(labelList[i]) + "<p>{}</p>".format("How does the firm will notify its user about privacy change?")
      
        if labelList[i] == "User Choice/Control":
          labelList[i] = "<h4>{}</h4>".format(labelList[i]) + "<p>{}</p>".format("What choices does the website provide to their users")
      
        if labelList[i] == "Privacy contact information":
          labelList[i] = "<h4>{}</h4>".format(labelList[i]) + "<p>{}</p>".format("Whom to contact in regards to privacy concerns.")
      




      finalResult = {}


      finalResult["report"] = result
      finalResult["label"]  = label
      finalResult["labelList"]  = labelList
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(exc_type, fname, exc_tb.tb_lineno)
      finalResult = {}


      finalResult["report"] = {"text": {
          "0": "Under construction"},
              "Label": {
                  "0": "Under construction"  
              }}
      finalResult["label"]  = "To be done"
      finalResult["labelList"]  = ["<h4>{}</h4>".format("ML Model Still under training for this data")]


    # result = {}
    response = app.response_class(response=dumps(finalResult),mimetype='application/json')
    return response


if __name__ == "__main__":
  app.run(port=5000, debug=True,host="0.0.0.0")




