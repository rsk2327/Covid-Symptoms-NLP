import pandas as pd
import numpy as np

import os

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 

from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM

from simpletransformers.language_modeling import LanguageModelingModel

from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

from tqdm import tqdm
import torch

import networkx as nx

import matplotlib.pyplot as plt


import plotly.graph_objects as go
from functools import partial

import pickle

from collections import deque

stop_words = set(stopwords.words('english')) 



from utils import *
# from plotting import *

import marshal

from ADRModel import *




model = BertForSequenceClassification.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1', output_hidden_states= True)

tokenizer = BertTokenizer.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1')


covidData = '/data1/roshansk/covid_data/'
df = pd.read_csv(os.path.join(covidData, 'messages_cm_mar1_apr23_noRT.csv'), nrows = 150000)

df = df[['message_id','user_id','message']]




# embList, msgList = getSymptomEmbedding(model,tokenizer, df, 'cough',0)
# indexList = [0,2,3,12,13,25]
# meanEmb = np.array(embList)[indexList,:]
# meanEmb = np.mean(meanEmb,0)


file = 'cough_19340_Emb.npy'
embList = np.load(os.path.join('EmbFolder/',file))
meanEmb = np.mean(embList,0)



graph = Graph()

graph.addNode('cough',0,0)
graph['cough'].vector = meanEmb

outputFolder = '/data1/roshansk/ADRModel_DataStore/'
modelFolder = './ModelFolder/Covid_cough_5_7/'

q = deque()
q.append(('cough',0))

# ADR = ADRModel(df, model, tokenizer, graph, outputFolder, q,  useMasterEmb=True, masterContrib=0.4, numThreshold = 100000)
ADR = ADRModel(df, model, tokenizer, graph, outputFolder, modelOutputFolder = modelFolder, 
               queue = q,  useMasterEmb=True, masterContrib=0.4, numThreshold=150000, saveEveryDepth= True)


print("Training started")

ADR.trainModel(maxDepth=5,topk=7) 

print("Training finished")


# ADR.model = None
# ADR.tokenizer = None
# ADR.df = None
      
# pickle.dump(ADR, open('Covid_fever_3_5_0.4.pkl','wb'))

print("Model saved")