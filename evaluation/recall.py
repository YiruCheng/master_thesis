# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import csv
   
class Recall:
    def __init__(self, output_file):
        self.output_file = output_file
        self.recall_list = {}
        self.doc_recall = {}
        
    def calculateRecall_k(self, topic, doc, candidate_entities, k):
        clean_topic = topic.replace('_', ' ').lower()
        if clean_topic in list(candidate_entities.keys())[:k]:
            self.doc_recall[doc] = 1.0
        
    def evaluate(self, topic, docs_size):
        if len(self.doc_recall) > 0:
            print(sum(self.doc_recall.values())/docs_size)
            self.recall_list[topic] = sum(self.doc_recall.values())/docs_size
        else:
            self.recall_list[topic] = 0
        
        self.doc_recall.clear()
        
    def outputResult(self):
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.recall_list.items())
        f.close()