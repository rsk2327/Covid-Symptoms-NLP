###############################
### Extract Symptoms 
### VERSION 2
### PyTorch GPU 
###############################

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


from tqdm import tqdm
import torch
import time
import pickle

from dask.distributed import Client

import torch
from torch.nn import CosineSimilarity

from functools import partial
from itertools import *

import itertools


device='cuda:2'


def getSimilarWords(tokenizer, combinedOutputFolder, symptom, meanEmb, similarityThreshold = 0.3, numThreshold = 150000, numComp = 10000):
        
    output = []

    symptomToken = tokenizer.encode(symptom)[1]

    fileList = os.listdir(combinedOutputFolder)
    
    cos = CosineSimilarity(dim=1, eps=1e-6)

    examineCount = 0
    

    for i in tqdm(range(len(fileList))):

        if examineCount >= numThreshold:
            break


        filename = os.path.join(combinedOutputFolder, f"{i}.pkl")
        subDict = pickle.load(open(filename,'rb'))

        IDList = subDict['id']
        tokenList = subDict['token']
        embList = subDict['emb']
        
        arrA = torch.from_numpy(meanEmb.reshape(1,-1)).to(device).type(torch.cuda.FloatTensor)
        arrB = torch.from_numpy(embList).to(device).type(torch.cuda.FloatTensor)
        
#         arrA = torch.from_numpy(meanEmb.reshape(1,-1)).to(device)
#         arrB = torch.from_numpy(embList).to(device)
        
        sim = cos(arrA,arrB).cpu().numpy().reshape(-1)
        
        del arrA
        del arrB
        
        sim = np.round(sim,4)

        index= np.where([sim> similarityThreshold])[1]

        tokenList_ = tokenList[index]
        IDList_ = IDList[index]
        simList = sim[index]

        out = [(x,y,z) for x,y,z in zip(tokenList_, simList, IDList_)]
        print(len(out))
        

        output += out

        examineCount += numComp
        

    return output



########################


# 'fever_9016_Emb.npy',
#  'fatigue_16342_Emb.npy',
#  'cough_19340_Emb.npy'


combinedOutputFolder = '/data2/roshansk/ADRModel_DataStore_10000/'
numComp = 10000
numThreshold = 1000000
file = 'cough_19340_Emb.npy'
symptom = ''
dumpFile = '/data1/roshansk/SymptomAnalysis/cough_1000k_thresh0.3_v2.p'

########################


dataFolder = '/data1/roshansk/covid_data/'
fileList = os.listdir(dataFolder)

df = pd.read_csv(os.path.join(dataFolder, fileList[0]), nrows = numThreshold)

tokenizer = BertTokenizer.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1')





embList = np.load(os.path.join('EmbFolder/',file))
meanEmb = np.mean(embList,0)


startTime = time.time()

result = getSimilarWords(tokenizer, combinedOutputFolder, symptom, 
                meanEmb, similarityThreshold = 0.3, numThreshold = numThreshold, numComp = numComp)

print(len(result))

print(f"Time taken : {time.time() - startTime}")


pickle.dump( result, open( dumpFile, "wb" ) )

print(f"Saved data for {file}")





