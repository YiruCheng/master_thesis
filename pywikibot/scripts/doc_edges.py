# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:06:50 2017

@author: Yiru
"""

import os
import csv
import pywikibot
import numpy

site = pywikibot.Site()

server_dir = '/home/mosman/work/Thesis/ThesisProject/ProjectData/TopicData/2016_2015'
tfidf_dir = '/tfidf'

def getTopKEntitiesByTFIDFperDoc(topic, doc, k):
    doc_dir = os.path.join(tfidf_dir, topic, doc)
    
    tfidf_map = {}
    with open(doc_dir, 'r', encoding='utf-8-sig') as f:
        content = f.readlines()
    f.close()
    for l in content:
        (entity, score) = l.split('|')
        if entity not in tfidf_map:
            tfidf_map[entity] = []
            tfidf_map[entity].append(float(score.strip('\n')))
            
    tfidf_map = sorted({k.replace(' ', '_').lower():numpy.sum(v) for k,v in tfidf_map.items()}.items(), key=lambda x:x[1], reverse=True)
    """top 200 by tfidf"""
    return dict(tfidf_map[:k])

def getEntitiesByFreq(topic_dir, doc):
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
    return entities


for topic in os.listdir(server_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(server_dir, topic, 'tagMe')
    
    for doc in os.listdir(topic_dir):
        entities = {}
        """
        entities = getTopKEntitiesByTFIDFperDoc(topic, doc, 10)
        """
        entities = getEntitiesByFreq(topic, doc)
        
        entity_pages = {k:pywikibot.Page(site, k) for k in entities if k}
        entity_edges = {}
        for k,p in entity_pages.items():
            entity_edges[k] = []
            outPage = p.linkedPages()
            for rest_entity in entities.keys():
                for out_p in outPage:
                    if rest_entity in out_p.title().replace('_', ' ').lower() and rest_entity != k and rest_entity not in entity_edges[k]:
                        entity_edges[k].append(rest_entity)
                    
        
        if not os.path.exists(os.path.join('../doc_edges', topic)):
            os.makedirs(os.path.join('../doc_edges', topic))
        
        with open(os.path.join('../doc_edges', topic, doc), 'w', newline='', encoding='utf-8-sig') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(entity_edges.items())
