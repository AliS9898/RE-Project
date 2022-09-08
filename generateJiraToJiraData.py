import random

from sentence_transformers import SentenceTransformer, util
import xml.etree.ElementTree as ET
import csv

model = SentenceTransformer('all-MiniLM-L6-v2')

jiraTree = ET.parse('data/jira.xml')
jiraRoot = jiraTree.getroot()

summaryList = []
for item in jiraTree.iter('item'):
    summary = item.find('summary').text
    summaryList.append(summary)


sentences = []

i = 0
while i < summaryList.__len__():
    sentences.append(summaryList[i])
    i += 1


#Compute embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.cos_sim(embeddings, embeddings)

#Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

# Sort scores in decreasing order
# pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

selectedPairs = []
selectedPairs = random.choices(pairs, k=10000)

num = 0
for pair in selectedPairs[0:10000]:
    i, j = pair['index']
    pair['type'] = 1
    print("NUM:", num, "{} \t\t {} \t\t Score: {:.4f} \t\t type: {}".format(sentences[i], sentences[j], pair['score'], pair['type']))
    num += 1

file = open('trainingDataSet.csv', 'w', newline='')

with file:
    #identifying headers
    header = ['Sentence1', 'Sentence2', 'Similarity Point', 'Similarity Type']
    writer = csv.DictWriter(file, fieldnames=header)

    #writing data row-wise into the csv file
    writer.writeheader()
    for pair in selectedPairs[0:10000]:
        i, j = pair['index']
        pair['type'] = 1
        writer.writerow({'Sentence1': sentences[i],
                         'Sentence2': sentences[j],
                         'Similarity Point': pair['score'].item() * 2,
                         'Similarity Type': pair['type']
                         })



#
# sentences2 = []
#
# j = 0
# while (j < 100):
#     index = random.randint(0, summaryList.__len__())
#     sentences2.append(summaryList[index])
#     j += 1
#
# embeddings1 = model.encode(sentences1, convert_to_tensor=True)
# embeddings2 = model.encode(sentences2, convert_to_tensor=True)
#
# cosine_scores = util.cos_sim(embeddings1, embeddings2)
#
# num = 0
# for i in range(len(sentences1)):
#     print(num, ":", "{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
#     num += 1