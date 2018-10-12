# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import os
import csv
from collections import OrderedDict
import pywikibot

site = pywikibot.Site()

dataset_dir = '/home/mosman/work/Thesis/ThesisProject/ProjectData/TopicData/2016_2015'
wiki_dir = '../wikipedia'


if not os.path.exists(wiki_dir):
    os.makedirs(wiki_dir)

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

        entities = OrderedDict(sorted(entities.items(), key=lambda x: x[1], reverse=True))
        entity_pages = {k:pywikibot.Page(site, k) for k in entities if k}
        wiki_entities = {}
        for key, page in entity_pages.items():
            if key in wiki_entities.keys():
                curr_freq = wiki_entities[key]
                wiki_entities[key] = curr_freq+1
            else:
                wiki_entities[key] = 1
                
            outPage = page.linkedPages()
            for out_p in outPage:
                clean_title = out_p.title().replace('_', ' ').lower()
                if clean_title in wiki_entities.keys():
                    curr_freq = wiki_entities[clean_title]
                    wiki_entities[clean_title] = curr_freq+1
                else:
                    wiki_entities[clean_title] = 1

        if not os.path.exists(os.path.join(wiki_dir, topic)):
            os.makedirs(os.path.join(wiki_dir, topic))
        
        with open(os.path.join(wiki_dir, topic, doc), 'w', newline='', encoding='utf-8-sig') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(wiki_entities.items())
        outfile.close()