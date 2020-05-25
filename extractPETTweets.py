import pandas as pd
import numpy as np
import os

import torch
import torchvision



import transformers
import simpletransformers

from simpletransformers.language_modeling import LanguageModelingModel
from simpletransformers.classification import ClassificationModel


from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM

from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils import *



model = ClassificationModel('bert', 'PETModel/checkpoint-2176-epoch-2/', num_labels=2, use_cuda=True) 



### Predicting on covid data

dataFolder = '/data1/roshansk/covid_data/'
covidDf = pd.read_csv(os.path.join(dataFolder, 'messages_cm_mar1_apr23_noRT.csv'))


covidDf = covidDf[['message','message_id']]
covidDf.columns = ['text','labels']

covidDf.labels = len(covidDf)*[0]


pred = model.eval_model(covidDf)

pred = pred[1]
finalPred = np.argmax(pred,1)

pd.Series(finalPred).to_csv('finalPred.csv', index = False, index_label = False)

covidDf['finalPred'] = finalPred

covidDf = covidDf[covidDf.finalPred == 1]


filename = '/data1/roshansk/covid_data/messages_cm_mar1_apr23_noRT_PET.csv'

covidDf.to_csv(filename, index = False, index_label = False)