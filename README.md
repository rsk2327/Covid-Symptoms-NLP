# Covid-Symptoms-NLP

## File List
* Covid LanguageModeling.ipynb

To be used to train a BERT model finetuned on your specific dataset. Refer to the Language Modeling section within notebook

* extractSymptoms.py

To be used to establish the embedding for your starting seed word. Helps with identifying samples corresponding to the seed woord which can then used to 
build the seed word embedding

* generateStates.py

Code that generates state embeddings for all tokens in messages within your dataset. Also contains
code for aggregating messages level output files into bigger aggregated files

* ADRModel_v5.py

Defines the ADRModel class thats utilised for the training process. Dont have to run this file

* trainADR_Covid.py

Performs the overall training of the model(graph). 
