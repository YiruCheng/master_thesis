# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import os
import random
import sys
sys.path.append('..')
from evaluation import mean_avg_precision, recall 

dataset_dir = '../wikipedia'

eval_map = mean_avg_precision.MAP('wiki_baseline_map.csv')
k = 5
eval_recall = recall.Recall('wiki_baseline_recall@'+str(k)+'.csv')

for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic)

    wiki_freq_dict = {}
    doc_recall = {}
    for doc in os.listdir(topic_dir):
        
        entities = {}
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            content = f.readlines()
        for l in content:
            if l:
                (entity, freq) = l.rsplit(',', 1)
                entities[entity.lower()] = freq.rstrip('\n')
        
        keys = list(entities.keys())
        random.shuffle(keys)
        
        eval_map.calculatePrecision(topic, doc, entities)
        eval_recall.calculateRecall_k(topic, doc, entities, k)
        
    eval_map.evaluate(topic)
    eval_recall.evaluate(topic, len(os.listdir(topic_dir)))

eval_map.outputResult()
eval_recall.outputResult()