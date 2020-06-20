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

import time

from utils import *
from plotting import *

import marshal


covidData = '/data1/roshansk/covid_data/'
df = pd.read_csv(os.path.join(covidData, 'messages_cm_mar1_apr23_noRT.csv'), nrows = 500000)
df = df[['message_id','user_id','message']]

model = BertForSequenceClassification.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1', output_hidden_states= True)
tokenizer = BertTokenizer.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1')

outputFolder = '/data1/roshansk/ADRModel_DataStore/'
embeddingType = 'last4sum'


for i in tqdm(range(150000,300000)):
            
    if os.path.exists(os.path.join(outputFolder, str(i)+".msh")):
        continue


    tokens = tokenizer.encode(df.iloc[i]['message'].lower())
    decoded = tokenizer.decode(tokens).split(" ")
    logits, hidden_states = model(torch.Tensor(tokens).unsqueeze(0).long())

    hidden_states = torch.stack(hidden_states).squeeze(1).permute(1,0,2)


    if embeddingType == 'last4sum':
        embedding = torch.sum(hidden_states[:,9:13,:],1)
    elif embeddingType =='last4concat':
        embedding = hidden_states[tokenIndex,9:13,:].reshape(-1)
    elif embeddingType == 'secondlast':
        embedding = hidden_states[tokenIndex,-2,:]
    else:
        embedding = hidden_states[tokenIndex,-1,:]


    embedding = embedding.detach().cpu().numpy()

    marshal.dump(embedding.tolist(), open(os.path.join(outputFolder, str(i)+".msh"), 'wb'))
    
    
    
