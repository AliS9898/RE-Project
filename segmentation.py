import pickle

import nltk
import numpy as np
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

stopwords = nltk.corpus.stopwords.words("english")

with open("data/transcript.txt", 'r') as f:
    corpus = f.readlines()
    corpus = [v.rstrip() for v in corpus if v != '\n']

# corpus = []
# for text in corpus0:
#     sentences = nltk.sent_tokenize(text)
#     text_arr = []
#     for sentence in sentences:
#         tokens = [w.lower() for w in nltk.word_tokenize(sentence)]
#         tokens = [w for w in tokens if w not in stopwords and len(w) > 1]
#         text_arr.append(' '.join(tokens))
#     corpus.append('.'.join(text_arr))

#docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
#docs = docs[0:10]

topic_model = BERTopic(min_topic_size=3)
topics, probs = topic_model.fit_transform(corpus)

to_save = []
print("topics = {}, probs = {}".format(topics, probs))
for index in topic_model.get_topics():
    if index == -1:
        continue
    topic = topic_model.get_topic(index)
    repr = topic_model.get_representative_docs(index)
    topic_name = topic_model.get_topic_info(index)["Name"].tolist()[0]
    print("topic = {}, repr = {}".format(topic_name, repr))
    to_save.append({
        "topic_name": topic_name,
        "repr": repr
    })
np.save("data/segments.npy", np.array(to_save))

with open("data/model", 'wb') as f:
    pickle.dump(topic_model, f)
