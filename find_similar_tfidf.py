import pickle
import re
import xml.etree.ElementTree as ET

import nltk as nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer

stopwords = nltk.corpus.stopwords.words("english")


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


stemmer = SnowballStemmer("english")


def tokenizer(s):
    tokens = re.sub(r"[^a-zA-Z]{2,}", " ", s).lower().split()
    tokens = [w for w in tokens if w not in stopwords and len(w) > 1]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


tasks = parse_tasks('data/Jira.xml')

with open("data/model", 'rb') as f:
    topic_model = pickle.load(f)

summarized = np.load("data/summarized.npy", allow_pickle=True)

tasks_text = []
for task in tasks:
    # tasks_text.append(" ".join((task[0], task[1])))
    tasks_text.append(tokenizer(task[0]))

tfidf_vectorizer = TfidfVectorizer()
tfidf_tasks = tfidf_vectorizer.fit_transform(tasks_text)

for summaries in summarized:
    most_similars = []
    for summary in summaries['summary']:
        summary_vector = tfidf_vectorizer.transform([tokenizer(summary)])
        sims = cosine_similarity(summary_vector, tfidf_tasks).flatten()
        ids = np.argsort(sims)[-10:]
        ids = ids[sims[ids] > 0.4]
        if not len(ids):
            continue
        similarity = [sims[i] for i in ids][::-1]
        similar_tasks = [tasks[index] for index in ids][::-1]
        most_similars = most_similars + [(similar_tasks[i], similarity[i]) for i in range(len(similar_tasks))]
    print("\n\nFind similar issues for topic {} with text (summary) {}".format(summaries['topic_name'], summaries['summary']))
    if len(most_similars) == 0:
        print("\tNo similar issues found")
    else:
        most_similars = sorted(most_similars, key=lambda i:i[1], reverse=True)
        if len(most_similars) > 5:
            most_similars = most_similars[0:5]
        for most_similar in most_similars:
            print("\tSimilarity = {}, issue = {}, title = {}, text = {}".format(most_similar[1], most_similar[0][1], most_similar[0][0], most_similar[0][2]))