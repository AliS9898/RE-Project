import random
from sentence_transformers import SentenceTransformer, util
import csv
from pandas import *
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

# with open('twcs.csv', mode='r', encoding='UTF8') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     sentences = csv_reader['text'].tolist()
#     for row in sentences:
#         print(row)

data = read_csv('twcs.csv')

sentences = data['text'].tolist()

conversations = []
for sentence in sentences:
    sentence = re.sub("@\w*", "", sentence)
    conversations.append(sentence)

selectedSentences = []
selectedSentences = random.choices(conversations, k=500)

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
    pair['type'] = 2
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
        pair['type'] = 2
        writer.writerow({'Sentence1': selectedSentences[i],
                         'Sentence2': selectedSentences[j],
                         'Similarity Point': pair['score'].item(),
                         'Similarity Type': pair['type']
                         })

