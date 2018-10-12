# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import numpy
import os
import re
import gensim
import string

dataset_dir = '../../sample-en-dataset'
centroid_dir = 'WE_freq200_centroid'

model = gensim.models.KeyedVectors.load_word2vec_format('/home/fnanni/work/Resources/W2V/GoogleNews-vectors-negative300.bin', binary=True)
for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic, 'tagMe')
    
    if not os.path.exists(os.path.join(centroid_dir, topic)):
        os.makedirs(os.path.join(centroid_dir, topic))
    
    for doc in os.listdir(topic_dir):
        entities = {}
        entity_vec = {}
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
        
        entities = dict(sorted(entities.items(), key=lambda x: x[1], reverse=True)[:200])

        #get avg vector of each word of entity
        for e in entities.keys():
            w = re.findall(r"[\w']+", string.capwords(e))
            avg_vec_tmp = numpy.zeros(shape=300)
            for tmp in w:
                if tmp in model.wv.vocab:
                    avg_vec_tmp = sum(avg_vec_tmp, model.wv[tmp]) / 2
            entity_vec[e] = avg_vec_tmp

        #calculate centroid
        try:
            centric = sum(entity_vec[e] for e in entities.keys() if e in entity_vec) / len(entity_vec)
        except ZeroDivisionError:
            centric = numpy.zeros(shape=200)
        centric.dump(os.path.join(centroid_dir, topic, doc))
        
del model