# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:56:00 2018

@author: Yiru
"""

import os
import random
import sys
sys.path.append('..')
from evaluation import mean_avg_precision, recall

dataset_dir = '../../sample-en-dataset'

eval_map = mean_avg_precision.MAP('doc_baseline_map.csv')
k = 5
eval_recall = recall.Recall('doc_baseline_recall@'+str(k)+'.csv')


for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic, 'tagMe')
    
    doc_recall = {}
    for doc in os.listdir(topic_dir):
        entities = {}
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            content = [l.strip().lower() for l in f.read().split('|') if l.strip().lower()]
        f.close()
        
        #count entity freq
        for e in content:
            if e in entities.keys():
                curr_freq = entities[e]
                entities[e] = curr_freq+1
            else:
                entities[e] = 1
        
        keys = list(entities.keys())
        random.shuffle(keys)
        eval_map.calculatePrecision(topic, doc, entities)
        eval_recall.calculateRecall_k(topic, doc, entities, k)
        
    eval_map.evaluate(topic)
    eval_recall.evaluate(topic, len(os.listdir(topic_dir)))

eval_map.outputResult()
eval_recall.outputResult()
