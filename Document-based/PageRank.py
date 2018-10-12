# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import os
from collections import OrderedDict
import ast
import numpy
import csv
from package import pagerank
import sys
sys.path.append('..')
from evaluation import mean_avg_precision, recall 


edges_dir = '../doc_edges'

eval_map = mean_avg_precision.MAP('doc_pagerank_map.csv')
k = 5
eval_recall = recall.Recall('doc_pagerank_recall@'+str(k)+'.csv')
for topic in os.listdir(edges_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(edges_dir, topic)
    
    doc_recall = {}
    for doc in os.listdir(topic_dir):
        
        entities = {}
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            entities = {row[0]: ast.literal_eval(row[1]) for row in reader}
        f.close()
        
        index_list = list(entities.keys())
        graph = numpy.zeros((len(entities.keys()),len(entities.keys())))
        for key,val in entities.items():
            for e in val:
                graph[index_list.index(key)][index_list.index(e)] = 1
            
        #print(graph)
        results = {}
        if len(graph) > 0:
            pr = pagerank.powerIteration(graph)
            for s in pr.iteritems():
                results[index_list[s[0]]] = s[1]
                #print(index_list[s[0]])
        
        results = OrderedDict(sorted(results.items(), key=lambda x: x[1], reverse=True))

        eval_map.calculatePrecision(topic, doc, results)
        eval_recall.calculateRecall_k(topic, doc, results, k)
        
    eval_map.evaluate(topic)
    eval_recall.evaluate(topic, len(os.listdir(topic_dir)))

eval_map.outputResult()
eval_recall.outputResult()