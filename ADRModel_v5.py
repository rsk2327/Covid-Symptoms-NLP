###############################
### ADR Model
### VERSION 4
### PyTorch GPU for getSimilar
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
from plotting import *

import marshal


from itertools import *

from dask.distributed import Client


import torch
from torch.nn import CosineSimilarity

from functools import partial
from itertools import *

import itertools


device='cuda:1'



class ADRModel(object):
    
    def __init__(self, df, model, tokenizer, graph, outputFolder, combinedOutputFolder, modelOutputFolder = './', queue=None, useMasterEmb = False, 
                 masterContrib = 0.5, embeddingType='last4sum',
                 numThreshold= 10000, saveEveryDepth = False,
                numComp = 10000):
        
        self.df = df
        self.model = model
        self.tokenizer = tokenizer
        self.graph = graph
        self.outputFolder = outputFolder
        self.combinedOutputFolder = combinedOutputFolder
        self.embeddingType = embeddingType
        self.numThreshold = numThreshold
        self.saveEveryDepth = saveEveryDepth
        self.modelOutputFolder = modelOutputFolder
        self.numComp = numComp
        
        if queue is None:
            self.q = deque()
        else:
            self.q = queue
            
            
        self.masterEmb = None
        
        self.useMasterEmb = useMasterEmb
        self.masterContrib = masterContrib
        
        self.masterEmbList = []
        
        
        self.generateStates()
        
        
    def generateStates(self):
        
        
        for i in tqdm(range(len(self.df))):
            
            if os.path.exists(os.path.join(self.outputFolder, f"{i}.msh")):
                continue


            tokens = self.tokenizer.encode(self.df.iloc[i]['message'].lower())
            decoded = self.tokenizer.decode(tokens).split(" ")
            logits, hidden_states = self.model(torch.Tensor(tokens).unsqueeze(0).long())

            hidden_states = torch.stack(hidden_states).squeeze(1).permute(1,0,2)

            
            if self.embeddingType == 'last4sum':
                embedding = torch.sum(hidden_states[:,9:13,:],1)
            elif self.embeddingType =='last4concat':
                embedding = hidden_states[tokenIndex,9:13,:].reshape(-1)
            elif self.embeddingType == 'secondlast':
                embedding = hidden_states[tokenIndex,-2,:]
            else:
                embedding = hidden_states[tokenIndex,-1,:]
                    
                    
            embedding = embedding.detach().cpu().numpy()
            
            marshal.dump(embedding.tolist(), open(os.path.join(self.outputFolder, f"{i}.msh"), 'wb'))
        
        
        
        
    def getSymptomEmbedding(self, symptom, subset = None):
    
        embeddingList = []
        messageList = []

#         if subset is not None:
#             self.df = self.df.iloc[subset]

#         if type(df) == pd.Series:
#             self.df = pd.DataFrame(self.df).T

#         symptomToken = self.tokenizer.encode(symptom)[1]
        symptomToken = self.tokenizer.convert_tokens_to_ids(symptom)

        for i in range(len(self.df)):

            if symptomToken in self.tokenizer.encode(self.df.iloc[i]['message'].lower()):

                tokens = self.tokenizer.encode(self.df.iloc[i]['message'].lower())
                decoded = self.tokenizer.decode(tokens).split(" ")

                hidden_states = np.array(marshal.load( open(os.path.join(self.outputFolder, f"{i}.msh"), 'rb') ))

                try:
                    tokenIndex = tokens.index(symptomToken)
                except:
                    a= 1
                    continue

 
                embedding = hidden_states[tokenIndex,:]

                embeddingList.append(embedding)
                messageList.append(self.df.iloc[i]['message'].lower())

                if len(embeddingList)==30:
                    break



        return embeddingList, messageList
    
    
    
    
    def getSimilarWords(self, symptom, meanEmb, similarityThreshold = 0.3):
        
        output = []

        symptomToken = self.tokenizer.encode(symptom)[1]

        fileList = os.listdir(self.combinedOutputFolder)

        cos = CosineSimilarity(dim=1, eps=1e-6)
        
        examineCount = 0
        
        for i in tqdm(range(len(fileList))):

            if examineCount >= self.numThreshold:
                break


            filename = os.path.join(self.combinedOutputFolder, f"{i}.pkl")
            subDict = pickle.load(open(filename,'rb'))

            IDList = subDict['id']
            tokenList = subDict['token']
            embList = subDict['emb']


            arrA = torch.from_numpy(meanEmb.reshape(1,-1)).to(device).type(torch.cuda.FloatTensor)
            arrB = torch.from_numpy(embList).to(device).type(torch.cuda.FloatTensor)

            sim = cos(arrA,arrB).cpu().numpy().reshape(-1)

            del arrA
            del arrB

            sim = np.round(sim,4)

            index= np.where([sim> similarityThreshold])[1]

            tokenList_ = tokenList[index]
            IDList_ = IDList[index]
            simList = sim[index]

            out = [(x,y,z) for x,y,z in zip(tokenList_, simList, IDList_)]

            output += out

            examineCount += self.numComp


        return output
        
        
    
    
    def getOutput(self, out):
    
        output = out
        

        outMap = {}

        for i in range(len(output)):
            if output[i][0] in outMap:
                outMap[output[i][0]].append(output[i][1])
            else:
                outMap[output[i][0]] = [output[i][1]]


        outMap_ = {}

        for i in range(len(output)):
            if output[i][0] in outMap_:
                outMap_[output[i][0]].append(output[i][2])
            else:
                outMap_[output[i][0]] = [output[i][2]]


        outputDf = []

        for key in outMap.keys():
            length = len(outMap[key])
            mean = np.mean(outMap[key])

            outputDf.append([key, length, mean])
            
    
        outputDf = pd.DataFrame(outputDf)
        outputDf.columns = ['word','counts','mean_sim']
        outputDf = outputDf.sort_values('mean_sim', ascending=False)

        return outputDf, outMap, outMap_
    
    
    
    
    def exploreNode(self, word, depth, maxDepth = 3, topk = 5):

    
        self.graph.addNode(word,0,depth)

        print(f"Depth : {depth} Exploring {word}")

        if depth == maxDepth:
            print("Reached max depth")
            return

        keyWord = word

        token = self.tokenizer.encode(keyWord)[1]

        if self.graph[word].vector is None:

            inEdgeList = self.graph[word].edges_in

            if len(inEdgeList)==0:
                textIDList = None
            else:
                textIDList = []

                for edge in inEdgeList:
                    textIDList.append(self.graph.edgeList[edge].textID)

                textIDList = list(set(list(itertools.chain.from_iterable(textIDList))))

            
            embList,msgList = self.getSymptomEmbedding(keyWord, subset = textIDList)

            meanEmb = np.array(embList)
            meanEmb = np.mean(meanEmb,0)


            self.graph[word].vector = meanEmb
            
            if self.masterEmb is None:
                self.masterEmb = meanEmb
            
            dist = getCosineDist(meanEmb, self.masterEmb)
            
            self.graph[word].masterDist = dist

        else:
            meanEmb = self.graph[word].vector
            
            if self.masterEmb is None:
                self.masterEmb = meanEmb
                
            dist = getCosineDist(meanEmb, self.masterEmb)
            
            self.graph[word].masterDist = dist


        symptom_ =''
        embList_ = meanEmb

        if self.useMasterEmb:
            
            finalEmb = self.masterContrib*self.masterEmb + (1 - self.masterContrib)*meanEmb
            
            out = self.getSimilarWords( symptom_, finalEmb , similarityThreshold = 0.3)
        else:
            out = self.getSimilarWords( symptom_, meanEmb, similarityThreshold = 0.3)
        
        
        outputDf, outMap, outMap_ = self.getOutput(out)

        outputDf = outputDf[outputDf.word!=keyWord]
    #     outputDf = outputDf[~outputDf.word.isin(list(graph.wordMap.keys()))]
        outputDf = outputDf.sort_values('mean_sim', ascending=False)
        outputDf = outputDf.head(topk)

        outputDf = outputDf[outputDf.mean_sim>0.4]

        print(outputDf)
        print("-----------------------")

        for i in range(len(outputDf)):

            word = outputDf.iloc[i]['word']
            numCount = outputDf.iloc[i]['counts']
            weight = outputDf.iloc[i]['mean_sim']
            textIDs = outMap_[word]

            wordList = set(self.graph.wordMap.keys())

            self.graph.addNode(word,0,depth+1)
            self.graph[word].textIDList.append(textIDs)
            self.graph.addEdge(keyWord, word, numCount, weight, textIDs)

            if word in wordList:
                continue

#             if "#" in word:
#                 continue


            self.q.append((word, depth+1))
            
            
    def trainModel(self, maxDepth = 3, topk = 5):
        
        currDepth = 0
        
        while len(self.q)>0:
            token, depth = self.q.popleft()
            
            if depth> currDepth:
                
                if self.saveEveryDepth:
                    filepath = os.path.join( self.modelOutputFolder, f"depth_{currDepth}.pkl")
                    self.saveModel(filepath)
                
                self.masterEmbList.append(self.masterEmb.copy())
                self.getMeanEmbedding(depth-1)
                currDepth += 1
            
            self.exploreNode(word = token, depth = depth, maxDepth=maxDepth, topk=topk)
        
        #Saving final model
        filepath = os.path.join(self.modelOutputFolder, "final.pkl")
        self.saveModel(filepath)


            
    def getMeanEmbedding(self, depth, topk = 3):
        
        candidates = self.graph.depthMap[depth]
        
        vals = [self.graph[x].masterDist for x in candidates]
        
        vals = [(x,y) for x,y in zip(candidates,vals)]
        
        vals = sorted(vals, key = lambda x : -x[1])
        
        meanEmb = self.masterEmb
        
        selectedWords = []
        for i in range(min(topk, len(vals)) ):
            meanEmb += self.graph[ vals[i][0] ].vector
            selectedWords.append(vals[i][0])
            
        meanEmb = meanEmb/(topk+1)
        
        self.masterEmb = meanEmb
        
        for i in range(len(selectedWords)):
            print(selectedWords[i])
        print("Master Embedding updated.")
        print("-----------------")
        
        
    
    def plotGraph(self):
        
        edgeList, nodeList, nodeValues, nodeCount, nodeText, nodeSize = getGraphComponents(self.graph)

        G=nx.Graph()

        G.add_nodes_from(nodeList)
        G.add_edges_from(edgeList)

        edge_trace, node_trace1, node_trace = getPlotlyComponents(G, nodeList, nodeSize, nodeValues, nodeText)


        fig = go.Figure(data=[edge_trace, node_trace1, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=50),

                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        
        fig.show()
        
        
    def saveModel(self,filename):
        
        classDict = self.__dict__.copy()
        classDict.pop('model')
        classDict.pop('tokenizer')
        classDict.pop('df')
        
        pickle.dump( classDict, open( filename, "wb" ) )
        
        
    def loadModel(self, filename):
        
        classDict = pickle.load(open(filename, 'rb'))
        
        for key in list(classDict.keys()):
            self.__dict__[key] = classDict[key]
        
        

        
def computeTask_(index, symptom, combinedOutputFolder,meanEmb, similarityThreshold):

    symptomToken = tokenizer.encode(symptom)[1]

    cos = CosineSimilarity(dim=1, eps=1e-6)

    filename = os.path.join(combinedOutputFolder, f"{index+6}.pkl")
    subDict = pickle.load(open(filename,'rb'))

    IDList = subDict['id']
    tokenList = subDict['token']
    embList = subDict['emb']

#         sim = np.round(cosine_similarity(embList, meanEmb.reshape(1,-1)).reshape(-1),4)

    arrA = torch.from_numpy(meanEmb.reshape(1,-1))
    arrB = torch.from_numpy(embList)

#         arrA = torch.from_numpy(meanEmb.reshape(1,-1)).cuda()
#         arrB = torch.from_numpy(embList).cuda()

    sim = cos(arrA,arrB).cpu().numpy().reshape(-1)

    sim = np.round(sim,4)

    index= np.where([sim> similarityThreshold])[1]

    tokenList_ = tokenList[index]
    IDList_ = IDList[index]
    simList = sim[index]

    out = [(x,y,z) for x,y,z in zip(tokenList_, simList, IDList_)]

    return out





