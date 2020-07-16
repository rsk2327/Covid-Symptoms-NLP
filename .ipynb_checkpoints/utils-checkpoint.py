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
from functools import partial
import itertools

import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english')) 

from collections import deque

import plotly.graph_objects as go



def getCosineDist(x,y):
    
    if x.shape == (768,):
        x = x.reshape(1,-1)
    if y.shape == (768,):
        y = y.reshape(1,-1)
        
    dist = cosine_similarity(x,y)
    
    return dist



def getSymptomEmbedding(model, tokenizer, df, symptom, symptomToken,  embeddingType = 'last4sum', subset = None):
    
    embeddingList = []
    messageList = []
    
    if subset is not None:
        df = df.iloc[subset]
    
    if type(df) == pd.Series:
        df = pd.DataFrame(df).T
    
    symptomToken = tokenizer.encode(symptom)[1]
    
    for i in range(len(df)):
        
        if symptomToken in tokenizer.encode(df.iloc[i]['message'].lower()):
                 
            tokens = tokenizer.encode(df.iloc[i]['message'].lower())
            decoded = tokenizer.decode(tokens).split(" ")
            logits, hidden_states = model(torch.Tensor(tokens).unsqueeze(0).long())

            hidden_states = torch.stack(hidden_states).squeeze(1).permute(1,0,2)
            
            
            try:
                tokenIndex = tokens.index(symptomToken)
            except:
                a= 1
                continue
#                 print(df.iloc[i]['message'])
#                 print(tokens)

            
            
            
            if embeddingType == 'last4sum':
                embedding = torch.sum(hidden_states[tokenIndex,9:13,:],0)
            elif embeddingType =='last4concat':
                embedding = hidden_states[tokenIndex,9:13,:].reshape(-1)
            elif embeddingType == 'secondlast':
                embedding = hidden_states[tokenIndex,-2,:]
            else:
                embedding = hidden_states[tokenIndex,-1,:]
                
                
            embeddingList.append(embedding.detach().cpu().numpy())
            messageList.append(df.iloc[i]['message'].lower())
            
            if len(embeddingList)==30:
                break
                
            
            
    return embeddingList, messageList




def getSimilarWords(model, tokenizer, df, symptom, embList, similarityThreshold = 0.3, numThreshold = 10000):
    
     
    output = []
   
    
    symptomToken = tokenizer.encode(symptom)[1]
    
    for i in range(len(df)):
        
        if symptomToken in tokenizer.encode(df.iloc[i]['message'].lower()):
                 
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



def getOutput(out):
    
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







class Node(object):
    
    def __init__(self, word, token, ID, vector = None, depth = None ):
        
        self.word = word
        self.token = token
        self.ID = ID
        self.depth = depth
        
        self.edges_in = []
        self.edges_out = []
        
        self.textIDList = []
        
        self.vector = vector
        
        self.masterDist = None
        
        
        
    def addInEdge(self, ID):
        self.edges_in.append(ID)
    
    def addOutEdge(self, ID):
        self.edges_out.append(ID)
        
    def getNodeInCount(self,graph):
        
        nodeCount = 0
        for i in range(len(self.edges_in)):
            edgeID = self.edges_in[i]
            nodeCount += graph.edgeList[edgeID].numCount
            
        return nodeCount
        
        
    def getWeightList(self,graph):
        
        weightList = []
        for i in range(len(self.edges_in)):
            edgeID = self.edges_in[i]
            weightList.append(graph.edgeList[edgeID].weight)
            
        if weightList == []:
            return [1.0]   #Weight for starting node
        else:
            return weightList
    
    
    def getOutEdges(self,graph):
        
        for i in range(len(self.edges_out)):
            edgeID = self.edges_out[i]
            edge = graph.edgeList[edgeID]
            nodeb = edge.nodeb

            print(f"{nodeb.word} || {edge.numCount} || {edge.weight}")
            
            
    def getInEdges(self,graph):
        
        for i in range(len(self.edges_in)):
            edgeID = self.edges_in[i]
            edge = graph.edgeList[edgeID]
            nodea = edge.nodea

            print(f"{nodea.word} || {edge.numCount} || {edge.weight}")
        
            
            
        
        
class Edge(object):
    
    def __init__(self, nodeA, nodeB, ID, numCount, weight, textID):
        
        self.nodea = nodeA
        self.nodeb = nodeB
        self.ID = ID
        
        self.textID = textID
        
        self.numCount = numCount
        self.weight = weight
        

        
        
class Graph(object):
    
    def __init__(self):
        
        self.nodeList = []
        self.edgeList = []
        
        self.wordMap = {}
        
        self.depthMap = {}
        
        
    def __getitem__(self,word):
        ID = self.wordMap[word]
        
        return self.nodeList[ID]
        
        
    def addNode(self, word, token , depth):
        
        if word in self.wordMap:
            return
        else:
            node = Node(word, token, len(self.nodeList), depth = depth)
            
            self.wordMap[word] = len(self.nodeList)
            
            self.nodeList.append(node)
            
            if depth in self.depthMap:
                self.depthMap[depth].append(word)
            else:
                self.depthMap[depth] = [word]
            
            
    def addEdge(self, wordA, wordB, numCount, weight, textID):
        
        ID = len(self.edgeList)
        
        nodeaID = self.wordMap[wordA]
        nodebID = self.wordMap[wordB]
        
        edge = Edge(self.nodeList[nodeaID], self.nodeList[nodebID], ID, numCount, weight, textID)
        
        self.edgeList.append(edge)
        
        self.nodeList[nodeaID].addOutEdge(ID)
        self.nodeList[nodebID].addInEdge(ID)
        
    
    def getTextIDs(self, word):
        ID = self.wordMap[word]
        
        node = self.nodeList[ID]
        
        textIDs = node.textIDList
        
        return textIDs
    
    
    def describeNode(self, word):
        
        node = self.__getitem__(word)
        print(f"Exploring {word}")
        
        
        for edgeID in node.edges_in:
            edge = self.edgeList[edgeID]

            worda = edge.nodea.word
            edgeCount = edge.numCount
            edgeWeight = edge.weight
            textIDs = edge.textID

            print(f"{worda:10} -> {word:10} | {edgeCount} | {np.round(edgeWeight,3):6} | {textIDs}")
        
        print("-"*20)
        
        
        for edgeID in node.edges_out:
            edge = self.edgeList[edgeID]

            wordb = edge.nodeb.word
            edgeCount = edge.numCount
            edgeWeight = edge.weight
            textIDs = edge.textID

            print(f"{word:10} -> {wordb:10} | {edgeCount} | {np.round(edgeWeight,3):6} | {textIDs}")
        
        
            
            

        
        
        
        
        
        


def exploreNode(word, depth, q, fullDf, graph, model, tokenizer, maxDepth = 3, topk = 5):

    
    graph.addNode(word,0, depth)
    
    print(f"Depth : {depth} Exploring {word}")
    
    if depth == maxDepth:
        print("Reached max depth")
        return
    
    keyWord = word

    token = tokenizer.encode(keyWord)[1]
    
    if graph[word].vector is None:
        
        inEdgeList = graph[word].edges_in
        
        if len(inEdgeList)==0:
            textIDList = None
        else:
            textIDList = []

            for edge in inEdgeList:
                textIDList.append(graph.edgeList[edge].textID)

            textIDList = list(set(list(itertools.chain.from_iterable(textIDList))))

        embList,msgList = getSymptomEmbedding(model, tokenizer, fullDf, keyWord, token, embeddingType='last4sum', subset = textIDList)
    
        meanEmb = np.array(embList)
        meanEmb = np.mean(meanEmb,0)
        
        
        graph[word].vector = meanEmb
        
    else:
        meanEmb = graph[word].vector
    
    
    symptom_ =''
    embList_ = meanEmb
    
    
    out = getSimilarWords(model, tokenizer, fullDf.iloc[0:100], symptom_, meanEmb, similarityThreshold = 0.3, numThreshold = 100000)
    
    outputDf, outMap, outMap_ = getOutput(out)
    
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
        
        wordList = set(graph.wordMap.keys())
        
        graph.addNode(word,0, depth+1)
        graph[word].textIDList.append(textIDs)
        graph.addEdge(keyWord, word, numCount, weight, textIDs)
        
        if word in wordList:
            continue
            
        if "#" in word:
            continue
            

        q.append((word, depth+1))
        
        
        
        
        
#         exploreNode(word,depth+1,q= q, fullDf = fullDf, graph = graph, model = model, tokenizer = tokenizer, maxDepth=maxDepth, topk=topk)
        
    