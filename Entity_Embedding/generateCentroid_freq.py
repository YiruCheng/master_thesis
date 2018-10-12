# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import numpy
import gensim
import os
from collections import OrderedDict


dataset_dir = '../../sample-en-dataset'
centroid_dir = 'EE_freq200_centroid'

model = gensim.models.Word2Vec.load('/home/mosman/work/Thesis/ThesisProject/ProjectDataLibs/rdf2vec/rdf2vec')
for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic, 'tagMe')
    
    if not os.path.exists(os.path.join(centroid_dir, topic)):
        os.makedirs(os.path.join(centroid_dir, topic))
    
    for doc in os.listdir(topic_dir):
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            content = [l.strip() for l in f.read().split('|') if l.strip()]
        f.close()
        
        entities = {}
        for e in content:
            if e in entities.keys():
                curr_freq = entities['dbr:'+e.replace(' ', '_')]
                entities['dbr:'+e.replace(' ', '_')] = curr_freq+1
            else:
                entities['dbr:'+e.replace(' ', '_')] = 1
        
        entities = OrderedDict(sorted(entities.items(), key=lambda x: x[1], reverse=True)[:200])

        #calculate centroid vector
        try:
            centric = sum(model.wv[e] for e in entities.keys() if e in model) / len([model.wv[e] for e in entities.keys() if e in model])
        except ZeroDivisionError:
            centric = numpy.zeros(shape=200)
        centric.dump(os.path.join(centroid_dir, topic, doc))
        
del model