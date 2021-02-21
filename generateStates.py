################################################
# This code processes your messages and saves the embeddings of tokens in a message as separate files. These embeddings of tokens # is later used during the iterative search process to find tokens similar to the context embedding. This code has 2 parts
# 1. Generates embeddings of each message individually by running it through the BERT model. This results in a single saved file 
# for each message
# 2. Collates individual files (typically 10000) from the first step into bigger files. This is done to speed up the iterative 
# search process
################################################


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
df = pd.read_csv(os.path.join(covidData, 'messages_cm_mar1_apr23_noRT.csv'), nrows = 1000000)
df = df[['message_id','user_id','message']]

model = BertForSequenceClassification.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1', output_hidden_states= True)
tokenizer = BertTokenizer.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1')

outputFolder = '/data1/roshansk/ADRModel_DataStore/'
embeddingType = 'last4sum'


###########################
######### PART 1 ##########
###########################


# device = 'cuda:1'

# model = model.to(device)

# startTime = time.time()

# for i in tqdm(range(850000,1000000)):
            
#     if os.path.exists(os.path.join(outputFolder, str(i)+".msh")):
#         continue


#     tokens = tokenizer.encode(df.iloc[i]['message'].lower())
#     decoded = tokenizer.decode(tokens).split(" ")
#     logits, hidden_states = model(torch.Tensor(tokens).to(device).unsqueeze(0).long())

#     hidden_states = torch.stack(hidden_states).squeeze(1).permute(1,0,2)


#     if embeddingType == 'last4sum':
#         embedding = torch.sum(hidden_states[:,9:13,:],1)
#     elif embeddingType =='last4concat':
#         embedding = hidden_states[tokenIndex,9:13,:].reshape(-1)
#     elif embeddingType == 'secondlast':
#         embedding = hidden_states[tokenIndex,-2,:]
#     else:
#         embedding = hidden_states[tokenIndex,-1,:]


#     embedding = embedding.detach().cpu().numpy()

#     marshal.dump(embedding.tolist(), open(os.path.join(outputFolder, str(i)+".msh"), 'wb'))
    

# print(f"Time taken : {time.time() - startTime}")
    
###########################
######### PART 2 ##########
###########################

def aggFiles(index, numComp, df, tokenizer, inputFolder, outputFolder):
    
    filename = os.path.join(outputFolder, f"{index}.pkl")
    print(filename)
    
    if os.path.exists(filename):
        print("Skipping file since it exists")
        return

    IDList = []
    tokenList = []
    embList = []

    for i in range(index*numComp, (index+1)*numComp):
        text = df.iloc[i]['message']

        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))


        emb = np.array(marshal.load(open(os.path.join(inputFolder, f"{i}.msh"),'rb' )))

        IDList += [i]*len(tokens)
        tokenList += tokens

        embList.append(emb)

    IDList = np.array(IDList)
    tokenList = np.array(tokenList)
    embList = np.concatenate(embList,axis=0)

    subDict = {'id':IDList, 'token':tokenList,'emb':embList}
        
    pickle.dump(subDict, open(filename,'wb'))
    
    
    
numComp = 10000
inputFolder = '/data1/roshansk/ADRModel_DataStore/'
outputFolder = '/data2/roshansk/ADRModel_DataStore_10000/'

for i in tqdm(range(90,100)):
    
    aggFiles(i, numComp, df, tokenizer, inputFolder, outputFolder)
        
    
################################################################################# 
    
    
    

    
    
    
