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

from functools import partial

import pickle

from utils import *

from collections import deque

stop_words = set(stopwords.words('english')) 




dataFolder = '/data1/roshansk/TwitterADR/'

df = pd.read_csv(os.path.join(dataFolder,'trainTweetData.csv'))
df1 = pd.read_csv(os.path.join(dataFolder,'train_tweet_annotations.tsv'),sep='\t', header = None)
df2 = pd.read_csv(os.path.join(dataFolder,'train_tweet_ids.tsv'),sep='\t', header = None)
df2.columns = ['id','user_id','text_id']
df1.columns = ['text_id','start','end','type','ADR','drug','drug1']
df1 = df1.merge(df2, on='text_id')
df = df1.merge(df[['id','text']], on='id')
trainDf = df.copy()

df = pd.read_csv(os.path.join(dataFolder,'testTweetData.csv'))
df1 = pd.read_csv(os.path.join(dataFolder,'test_tweet_annotations.tsv'),sep='\t', header = None)
df2 = pd.read_csv(os.path.join(dataFolder,'test_tweet_ids.tsv'),sep='\t', header = None)
df2.columns = ['id','user_id','text_id']
df1.columns = ['text_id','start','end','type','ADR','drug','drug1']
df1 = df1.merge(df2, on='text_id')
df = df1.merge(df[['id','text']], on='id')
testDf = df.copy()

fullDf = pd.concat([trainDf,testDf],axis =0)
fullDf['message'] = fullDf.text
fullDf.drop_duplicates(subset='message',inplace =True)


### Defining BERT Model
model = BertForSequenceClassification.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1', output_hidden_states= True)
tokenizer = BertTokenizer.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1')



graph = Graph()

q = deque()
q.append(('tired',0))

# exploreNode_ = partial(exploreNode, q= q, fullDf = fullDf, graph = graph, model = model, tokenizer = tokenizer, maxDepth=2, topk=6)

while len(q)>0:
    token, depth = q.pop()
    exploreNode(token, depth = depth, q= q, fullDf = fullDf, graph = graph, model = model, tokenizer = tokenizer, maxDepth=3, topk=6)
    
    
pickle.dump( graph, open( "tired_graph_3_6_textSubset.p", "wb" ) )
