import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv

df = pd.read_csv('trainingDataSet.csv', sep=",", encoding='unicode_escape')

# print(df['Sentence1'], "\n", df['Sentence2'], "\n", df['Similarity Point'], "\n", df['Similarity Type'])

similarity_points = df['Similarity Point']
similarity_score = []
for point in similarity_points:
    similarity_score.append([float(point)])

scaler = MinMaxScaler()
similarity_scaled = scaler.fit_transform(similarity_score)
for point in similarity_scaled:
    print(point)


file = open('data/trainingData.csv', 'w', newline='', encoding='utf-8')

with file:
    # identifying headers
    header = ['Sentence1', 'Sentence2', 'Similarity Point', 'Similarity Type']
    writer = csv.DictWriter(file, fieldnames=header)

    #writing data row-wise into the csv file
    writer.writeheader()
    i = 0
    print(df['Sentence1'][i])
    while i < len(similarity_scaled):
        writer.writerow({'Sentence1': df['Sentence1'][i],
                         'Sentence2': df['Sentence2'][i],
                         'Similarity Point': similarity_scaled[i].item(),
                         'Similarity Type': df['Similarity Type'][i]
                         })
        i += 1

