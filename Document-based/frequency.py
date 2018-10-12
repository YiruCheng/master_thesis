# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import os
from collections import OrderedDict
import sys
sys.path.append('..')
from evaluation import mean_avg_precision, recall 


dataset_dir = '../../sample-en-dataset'
eval_map = mean_avg_precision.MAP('doc_freq200_map.csv')
k = 5
eval_recall = recall.Recall('doc_freq200_recall@'+str(k)+'.csv')
for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic, 'tagMe')
    
    doc_recall = {}
    for doc in os.listdir(topic_dir):
        
        entities = {}
        
        """
        counting the frequency of each entity
        """
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            content = [l.strip().lower() for l in f.read().split('|') if l.strip().lower()]
        f.close()
        for e in content:
            if e in entities.keys():
                curr_freq = entities[e]
                entities[e] = curr_freq+1
            else:
                entities[e] = 1

        entities = OrderedDict(sorted(entities.items(), key=lambda x: x[1], reverse=True)[:200])
        """ all entities sorted by frequency
        entities = OrderedDict(sorted(entities.items(), key=lambda x: x[1], reverse=True))
        """
        
        eval_map.calculatePrecision(topic, doc, entities)
        eval_recall.calculateRecall_k(topic, doc, entities, k)
        
    eval_map.evaluate(topic)
    eval_recall.evaluate(topic, len(os.listdir(topic_dir)))

eval_map.outputResult()
eval_recall.outputResult()