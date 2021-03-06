# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import os
from collections import OrderedDict
import sys
sys.path.append('..')
from evaluation import mean_avg_precision, recall 


dataset_dir = '../wikipedia'

eval_map = mean_avg_precision.MAP('wiki_freq200_map.csv')
k = 5
eval_recall = recall.Recall('wiki_freq200_recall@'+str(k)+'.csv')
for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic)
    
    doc_recall = {}
    for doc in os.listdir(topic_dir):
        
        entities = {}
        
        """
        counting the frequency of each entity
        """
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            content = f.readlines()
        f.close()
        for l in content:
            if l:
                (entity, freq) = l.rsplit(',', 1)
                entities[entity.lower()] = freq.rstrip('\n')

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