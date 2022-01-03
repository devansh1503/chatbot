import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
from flask import Flask,render_template,request,jsonify

with open("intents.json") as file:
    data = json.load(file)

net = tflearn.input_data(shape=[None, 55])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,9,activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

model.load('model.tflearn')

words = []
labels=[]
docs_x=[]
docs_y=[]
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern) #splits the words in the sentences of patterns
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=(1)
    return numpy.array(bag)
def chatfun(inp):
    res = model.predict([bag_of_words(inp,words)])
    res_index = numpy.argmax(res)
    tag = labels[res_index]
    for tg in data["intents"]:
        if tg["tag"] == tag:
            responses = tg['responses']
    return random.choice(responses)

