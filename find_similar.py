import os.path
import pickle
import xml.etree.ElementTree as ET
import xlsxwriter

import numpy as np
from bertopic import BERTopic
from bertopic.backend._sentencetransformers import SentenceTransformerBackend
from sklearn.metrics.pairwise import cosine_similarity

samples = np.load("data/segments.npy", allow_pickle=True)
samples_dict = {}
for item in samples:
    samples_dict[item['topic_name']] = ';'.join(item['repr'])
#print(samples_dict)

def parse_tasks(path):
    tree = ET.parse(path)
    root = tree.getroot()

    tasks = []

    for item in root.findall("./channel/item"):
        title = item.find("./title").text
        title = title.replace("\n", "")
        link = item.find("./link").text
        summary = item.find("./summary").text
        priority = item.find("./priority").text
        tasks.append((title, link, summary, priority))

    return tasks


tasks = parse_tasks('data/Jira.xml')

with open("data/model", 'rb') as f:
    topic_model = pickle.load(f)

summarized = np.load("data/summarized.npy", allow_pickle=True)

tasks_text = []
for task in tasks:
    # tasks_text.append(" ".join((task[0], task[1])))
    tasks_text.append(task[0])

topic_model.embedding_model = SentenceTransformerBackend("all-MiniLM-L6-v2")
embedding_tasks = topic_model._extract_embeddings(tasks_text, method="document", verbose=topic_model.verbose)

corpus = []

xls_file = 'result_sim.xlsx'
if os.path.isfile(xls_file):
    os.remove(xls_file)

workbook = xlsxwriter.Workbook(xls_file)
worksheet = workbook.add_worksheet()

sheet_data = []
sheet_data.append([
    'Topic Name', 'Topic Representative', 'Summary', 'Most Similar Issue', 'Text of Issue', 'Similarity (Confidence)'
])

for summaries in summarized:
    most_similars = []
    for summary in summaries['summary']:
        embedding_requirements = topic_model._extract_embeddings([summary], method="document",
                                                                 verbose=topic_model.verbose)
        sims = cosine_similarity(embedding_requirements.reshape(1, -1), embedding_tasks).flatten()
        ids = np.argsort(sims)[-10:]
        ids = ids[sims[ids] > 0.4]
        if not len(ids):
            continue
        similarity = [sims[i] for i in ids][::-1]
        similar_tasks = [tasks[index] for index in ids][::-1]
        most_similars = most_similars + [(similar_tasks[i], similarity[i]) for i in range(len(similar_tasks))]
    summary_text = ';'.join(summaries['summary'])
    print("\n\nFind similar issues for topic {} with text (summary) {}".format(summaries['topic_name'],
                                                                               summaries['summary']))
    if len(most_similars) == 0:
        sheet_data.append([summaries['topic_name'], samples_dict[summaries['topic_name']], summary_text, 'no'])
        print("\tNo similar issues found")
    else:
        most_similars = sorted(most_similars, key=lambda i: i[1], reverse=True)
        if len(most_similars) > 5:
            most_similars = most_similars[0:5]
        for most_similar in most_similars:
            print("\tSimilarity = {}, issue = {}, title = {}, text = {}".format(most_similar[1], most_similar[0][1],
                                                                                most_similar[0][0], most_similar[0][2]))
            try:
                sheet_data.append(
                [summaries['topic_name'], samples_dict[summaries['topic_name']], summary_text, most_similar[0][1], most_similar[0][2], most_similar[1]])
            except Exception as ex:
                print(ex)

for i, row in enumerate(sheet_data):
    for j, item in enumerate(row):
        worksheet.write(i, j, item)

workbook.close()
