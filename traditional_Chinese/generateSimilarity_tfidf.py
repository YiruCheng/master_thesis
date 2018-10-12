# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import numpy
from scipy import spatial
import gensim
import os
import csv
from collections import OrderedDict


dataset_dir = '../../chinese-dataset/CKIP'
centroid_dir = 'EE_tfidf_centroid'
similarity_dir = 'EE_tfidf_similarity'
tfidf_dir = 'tfidf_zh_ckip'

model = gensim.models.KeyedVectors.load_word2vec_format('pre-trained_vectors/wiki.zh_classical.vec')

def getTopKEntitiesByTFIDFperDoc(topic, doc, k):
    doc_dir = os.path.join(tfidf_dir, topic, doc)
    
    tfidf_map = {}
    with open(doc_dir, 'r', encoding='utf-8-sig') as f:
        content = f.readlines()
    f.close()
    for l in content:
        (entity, score) = l.split('|')
        if entity not in tfidf_map:
            tfidf_map[entity] = []
            tfidf_map[entity].append(float(score.strip('\n')))
            
    tfidf_map = sorted({k.replace(' ', '_').lower():numpy.sum(v) for k,v in tfidf_map.items()}.items(), key=lambda x:x[1], reverse=True)
    return dict(tfidf_map[:k])


for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic)
    
    if not os.path.exists(os.path.join(similarity_dir, topic)):
        os.makedirs(os.path.join(similarity_dir, topic))
    
    for doc in os.listdir(topic_dir):
        entities = {}
        if len(os.listdir(topic_dir)) > 0:
            entities = getTopKEntitiesByTFIDFperDoc(topic, doc, 200)

        #calculate consine similarity
        with open(os.path.join(centroid_dir, topic, doc), 'rb') as f:
            f.seek(0)
            centroid = numpy.load(f)
        f.close()
        
        similarity_dict = {e: 1-spatial.distance.cosine(model[e], centroid) for e in entities.keys() if e in model}
        
        sorted_x = sorted(similarity_dict.items(), key=lambda x:x[1], reverse=True)
        with open(os.path.join(similarity_dir, topic, doc), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(sorted_x)
        f.close()
        
del model