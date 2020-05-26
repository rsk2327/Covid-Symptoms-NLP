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

import pickle

def getSimilarWords(model, df, symptom, embList, similarityThreshold = 0.3, numThreshold = 10000):
    
     
    output = []
    
    for i in tqdm(range(len(df))):
        
        if symptom in df.iloc[i]['message'].lower():
                 
            tokens = tokenizer.encode(df.iloc[i]['message'].lower())
            decoded = tokenizer.decode(tokens).split(" ")
            logits, hidden_states = model(torch.Tensor(tokens).unsqueeze(0).long())

            hidden_states = torch.stack(hidden_states).squeeze(1).permute(1,0,2)
            
            
            hidden_states = hidden_states[:,9:13,:]
            hidden_states = torch.sum(hidden_states,1).detach().cpu().numpy()
            
            similarity = cosine_similarity(hidden_states, embList.reshape(1,-1)).reshape(-1)

                            
            index = np.where([similarity> similarityThreshold])[1]

            selectTokens = np.array(tokens)[index]
            selectSim = similarity[index]
                      


            for j in range(len(index)):
                token = tokenizer.ids_to_tokens[selectTokens[j]]
                sim = selectSim[j]
                output.append((token, sim,i))

            
        if i==numThreshold:
            break
            
    return output




########################


# 'fever_9016_Emb.npy',
#  'fatigue_16342_Emb.npy',
#  'cough_19340_Emb.npy'

file = 'fatigue_16342_Emb.npy'
dumpFile = '/data1/roshansk/SymptomAnalysis/fatigue_10k_thresh0.3.p'
symptom = ''

########################


dataFolder = '/data1/roshansk/covid_data/'
fileList = os.listdir(dataFolder)

df = pd.read_csv(os.path.join(dataFolder, fileList[0]), nrows = 1500000)

model = BertForSequenceClassification.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1', output_hidden_states= True)

tokenizer = BertTokenizer.from_pretrained('/data1/roshansk/Exp1/checkpoint-141753-epoch-1')




embList = np.load(os.path.join('EmbFolder/',file))
embList = np.mean(embList,0)


out = getSimilarWords(model, df, symptom, embList, similarityThreshold = 0.3, numThreshold = 10000)





pickle.dump( out, open( dumpFile, "wb" ) )

print(f"Saved data for {file}")





