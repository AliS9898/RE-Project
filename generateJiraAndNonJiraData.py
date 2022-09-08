from sentence_transformers import SentenceTransformer, util
from pandas import *
import csv
import re
import random
import xml.etree.ElementTree as ET

model = SentenceTransformer('all-MiniLM-L6-v2')

data = read_csv('twcs.csv')

sentences = data['text'].tolist()

conversations = []
for sentence in sentences:
    sentence = re.sub("@\w*", "", sentence)
    conversations.append(sentence)

selectedSentences = []
selectedSentences = random.choices(conversations, k=500)

jiraTree = ET.parse('data/jira.xml')
jiraRoot = jiraTree.getroot()

for item in jiraTree.iter('item'):
    summary = item.find('summary').text
    selectedSentences.append(summary)

#Compute embeddings
embeddings = model.encode(selectedSentences, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.cos_sim(embeddings, embeddings)

#Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

selectedPairs = []
selectedPairs = random.choices(pairs, k=20000)

num = 0
for pair in selectedPairs[0:20000]:
    i, j = pair['index']
    pair['type'] = 3
    print("NUM:", num, "{} \t\t {} \t\t Score: {:.4f} \t\t type: {}".format(selectedSentences[i], selectedSentences[j], pair['score'], pair['type']))
    num += 1

file = open('trainingDataSet.csv', 'a', newline='', encoding='utf-8')

with file:
    # identifying headers
    header = ['Sentence1', 'Sentence2', 'Similarity Point', 'Similarity Type']
    writer = csv.DictWriter(file, fieldnames=header)

    #writing data row-wise into the csv file
    writer.writeheader()
    for pair in selectedPairs[0:20000]:
        i, j = pair['index']
        pair['type'] = 3
        writer.writerow({'Sentence1': selectedSentences[i],
                         'Sentence2': selectedSentences[j],
                         'Similarity Point': pair['score'].item(),
                         'Similarity Type': pair['type']
                         })