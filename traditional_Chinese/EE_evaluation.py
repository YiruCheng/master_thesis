# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import os
from collections import OrderedDict
import sys
import csv
sys.path.append('..')
from evaluation import mean_avg_precision, recall 


similarity_dir = 'EE_freq200_similarity'

eval_map = mean_avg_precision.MAP('zh_CKIP_EE_freq200_map.csv')
k = 5
eval_recall = recall.Recall('zh_CKIP_doc_freq200_recall@'+str(k)+'.csv')
for topic in os.listdir(similarity_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(similarity_dir, topic)
    
    doc_recall = {}
    for doc in os.listdir(topic_dir):
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            similarity_dict = OrderedDict({l[0] :l[1] for l in reader}.items())
        f.close()
        
        eval_map.calculatePrecision(topic, doc, similarity_dict)
        eval_recall.calculateRecall_k(topic, doc, similarity_dict, k)
        
    eval_map.evaluate(topic)
    eval_recall.evaluate(topic, len(os.listdir(topic_dir)))

eval_map.outputResult()
eval_recall.outputResult()