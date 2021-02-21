import pandas as pd, numpy as np, os, nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM
from simpletransformers.language_modeling import LanguageModelingModel
from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from tqdm import tqdm
import torch, networkx as nx
from functools import partial
import matplotlib.pyplot as plt
stop_words = set(stopwords.words('english'))
from collections import deque
import plotly.graph_objects as go
from networkx.drawing.nx_agraph import graphviz_layout

def getPlotlyComponents(G, nodeList, nodeSize, nodeValues, nodeText):
    pos = graphviz_layout(G, prog='twopi', args='')
    for n, p in pos.items():
        G.nodes[n]['pos'] = p

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x,
      y=edge_y,
      line=dict(width=0.5, color='#888'),
      hoverinfo='none',
      mode='lines')
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace1 = go.Scatter(x=node_x,
      y=node_y,
      mode='markers+text',
      text=nodeList,
      textposition='bottom center',
      textfont= dict(size=16),
      marker=dict(showscale=False,
      colorscale='YlGnBu',
      reversescale=True,
      color=[],
      size=nodeSize,
      line_width=0))
    node_trace = go.Scatter(x=node_x,
      y=node_y,
      mode='markers',
      hoverinfo='text',
      marker=dict(showscale=True,
      colorscale='Reds',
      reversescale=False,
      color=[],
      size=nodeSize,
      colorbar=dict(thickness=15,
      title='Node Connections',
      xanchor='left',
      titleside='right'),
      line_width=1))
    node_trace.marker.color = nodeValues
    node_trace.text = nodeText
    return (
     edge_trace, node_trace1, node_trace)


def getGraphComponents(graph, sizeMult=10, sizeConstant=5):
    sizeMult = 10
    sizeConstant = 5
    edgeList = []
    for i in range(len(graph.edgeList)):
        edge = graph.edgeList[i]
        edgeList.append((edge.nodea.word, edge.nodeb.word))

    nodeList = []
    nodeValues = []
    nodeCount = []
    nodeText = []
    for i in range(len(graph.nodeList)):
        node = graph.nodeList[i]
        nodeList.append(node.word)
        val = np.round(np.mean(node.getWeightList(graph)), 3)
        count = node.getNodeInCount(graph)
        nodeValues.append(val)
        nodeCount.append(count)
        text = f"{node.word} <br>Count : {count} <br>Score : {val}"
        nodeText.append(text)

    nodeSize = list((MinMaxScaler().fit_transform(np.array(nodeCount).reshape(-1, 1)) * sizeMult + sizeConstant).reshape(-1))
    
    return (edgeList, nodeList, nodeValues, nodeCount, nodeText, nodeSize)