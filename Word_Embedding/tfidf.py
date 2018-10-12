# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import os
from collections import OrderedDict
import csv
import sys
sys.path.append('..')
from evaluation import embedding_map, embedding_recall 


similarity_dir = 'WE_tfidf_similarity'

eval_map = embedding_map.MAP('WE_tfidf_map.csv')
k = 5
eval_recall = embedding_recall.Recall('WE_tfidf_recall@'+str(k)+'.csv')
for topic in os.listdir(similarity_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(similarity_dir, topic)
    
    doc_recall = {}
    for doc in os.listdir(topic_dir):
        
        with open(os.path.join(topic_dir, doc), 'r') as f:
            reader = csv.reader(f)
            similarity_dict = OrderedDict({l[0].lower() :l[1] for l in reader}.items())
        f.close()
        
        q = topic.lower().replace('_', ' ')
        eval_map.calculatePrecision(q, doc, similarity_dict)
        eval_recall.calculateRecall_k(q, doc, similarity_dict, k)
        
    eval_map.evaluate(topic)
    eval_recall.evaluate(topic, len(os.listdir(topic_dir)))

eval_map.outputResult()
eval_recall.outputResult()