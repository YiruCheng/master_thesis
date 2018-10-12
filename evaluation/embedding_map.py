# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import csv

class MAP:
    
    def __init__(self, output_file):
        self.output_file = output_file
        self.mean_avg_precision = {}
        self.doc_recall = {}
        
    def calculatePrecision(self, topic, doc, candidate_entities):
        if topic in dict(candidate_entities):
            self.doc_recall[doc] = 1 / (list(candidate_entities.keys()).index(topic)+1)
        
    def evaluate(self, topic):
        if len(self.doc_recall) > 0:
            print(sum(self.doc_recall.values())/len(self.doc_recall))
            self.mean_avg_precision[topic] = sum(self.doc_recall.values())/len(self.doc_recall)
        else:
            self.mean_avg_precision[topic] = 0
            
        self.doc_recall.clear()
    
    def outputResult(self):
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.mean_avg_precision.items())
        f.close()
        