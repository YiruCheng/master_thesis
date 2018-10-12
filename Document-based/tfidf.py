# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import os
import numpy
import sys
sys.path.append('..')
from evaluation import mean_avg_precision, recall 


dataset_dir = '../../sample-en-dataset'
tfidf_dir = '/tfidf'

eval_map = mean_avg_precision.MAP('doc_tfidf_map.csv')
k = 5
eval_recall = recall.Recall('doc_tfidf_recall@'+str(k)+'.csv')

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
    
    doc_recall = {}
    for doc in os.listdir(topic_dir):
        
        entities = {}
        entities = getTopKEntitiesByTFIDFperDoc(topic, doc, 200)
        
        eval_map.calculatePrecision(topic, doc, entities)
        eval_recall.calculateRecall_k(topic, doc, entities, k)
        
    eval_map.evaluate(topic)
    eval_recall.evaluate(topic, len(os.listdir(topic_dir)))

eval_map.outputResult()
eval_recall.outputResult()