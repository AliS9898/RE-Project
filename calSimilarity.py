from sentence_transformers.SentenceTransformer import SentenceTransformer
import pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
import pandas as pd
import csv

filename = 'finalized_model50.sav'
loaded_model = pickle.load(open(filename, 'rb'))
model_preTrained = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('trainingData.csv', sep=",", encoding='unicode_escape')
X, y, z = df[['Sentence1','Sentence2']], df['Similarity Point'].astype('float32'), df['Similarity Type']

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=0.30, random_state=42)
# Two lists of sentences
sentences1 = []
for item in X_test['Sentence1']:
    sentences1.append(item)

sentences2 = []
for item in X_test['Sentence2']:
    sentences2.append(item)

similarity_type = []
for item in z_test:
  similarity_type.append(item)

#Compute embedding for both lists
embeddings1 = loaded_model.encode(sentences1, convert_to_tensor=True)
embeddings2 = loaded_model.encode(sentences2, convert_to_tensor=True)

embeddings1_preTrained = model_preTrained.encode(sentences1, convert_to_tensor=True)
embeddings2_preTrained = model_preTrained.encode(sentences2, convert_to_tensor=True)


#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)
consine_scores_preTrained = util.cos_sim(embeddings1_preTrained, embeddings2_preTrained)

#Output the pairs with their score
# for i in range(len(sentences1)):
#     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))

file = open('scores.csv', 'w', newline='')

with file:
    #identifying headers
    header = ['Sentence1', 'Sentence2', 'Cosine Score', 'Cosine Score Pre-Trained', 'Similarity Type']
    writer = csv.DictWriter(file, fieldnames=header)

    #writing data row-wise into the csv file
    writer.writeheader()
    for i in range(len(sentences1)):
      writer.writerow({'Sentence1': sentences1[i],
                      'Sentence2': sentences2[i],
                      'Cosine Score': cosine_scores[i][i].item(),
                      'Cosine Score Pre-Trained': consine_scores_preTrained[i][i].item(),
                      'Similarity Type': similarity_type[i]
      })