# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import os
from collections import OrderedDict
import sys
sys.path.append('..')
from evaluation import mean_avg_precision, recall 


dataset_dir = '../../chinese-dataset'
tfidf_dir = '/zh_tfidf'

eval_map = mean_avg_precision.MAP('zh_CKIP_doc_freq200_map.csv')
k = 5
eval_recall = recall.Recall('zh_CKIP_doc_freq200_recall@'+str(k)+'.csv')
for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic)
    
    doc_recall = {}
    for doc in os.listdir(topic_dir):
        entities = {}
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        f.close()
        
        for e in content:
            if e in entities.keys():
                curr_freq = entities[e]
                entities[e] = curr_freq+1
            else:
                entities[e] = 1
        entities = OrderedDict(sorted(entities.items(), key=lambda x: x[1], reverse=True))
        
        eval_map.calculatePrecision(topic, doc, entities)
        eval_recall.calculateRecall_k(topic, doc, entities, k)
        
    eval_map.evaluate(topic)
    eval_recall.evaluate(topic, len(os.listdir(topic_dir)))

eval_map.outputResult()
eval_recall.outputResult()