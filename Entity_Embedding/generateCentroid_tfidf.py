# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import numpy
import gensim
import os

dataset_dir = '../../sample-en-dataset'
tfidf_dir = 'tfidf'
centroid_dir = 'EE_tfidf_centroid'

model = gensim.models.Word2Vec.load('/home/mosman/work/Thesis/ThesisProject/ProjectDataLibs/rdf2vec/rdf2vec')

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
    topic_dir = os.path.join(dataset_dir, topic, 'tagMe')
    
    if not os.path.exists(os.path.join(centroid_dir, topic)):
        os.makedirs(os.path.join(centroid_dir, topic))
    
    for doc in os.listdir(topic_dir):
        entities = {}
        if len(os.listdir(topic_dir)) > 0:
            entities = getTopKEntitiesByTFIDFperDoc(topic, doc, 200)
            entities = {'dbr:'+e:score for e, score in entities.items()}
        
        #calculate centroid vector
        try:
            centric = sum(model.wv[e] for e in entities.keys() if e in model) / len([model.wv[e] for e in entities.keys() if e in model])
        except ZeroDivisionError:
            centric = numpy.zeros(shape=200)
        centric.dump(os.path.join(centroid_dir, topic, doc))
        
del model