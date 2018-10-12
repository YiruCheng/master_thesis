# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import numpy
from scipy import spatial
import gensim
import os
import re
from math import isnan
import string
import csv
from collections import OrderedDict


dataset_dir = '../../sample-en-dataset'
centroid_dir = 'WE_freq200_centroid'
similarity_dir = 'WE_freq200_similarity'

model = gensim.models.KeyedVectors.load_word2vec_format('/home/fnanni/work/Resources/W2V/GoogleNews-vectors-negative300.bin', binary=True)
for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic, 'tagMe')
    
    if not os.path.exists(os.path.join(similarity_dir, topic)):
        os.makedirs(os.path.join(similarity_dir, topic))
    
    for doc in os.listdir(topic_dir):
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            content = [l.strip() for l in f.read().split('|') if l.strip()]
        f.close()
        
        entities = {}
        entity_vec = {}
        for e in content:
            if e in entities.keys():
                curr_freq = entities[e]
                entities[e] = curr_freq+1
            else:
                entities[e] = 1
        
        entities = OrderedDict(sorted(entities.items(), key=lambda x: x[1], reverse=True)[:200])
        #get avg vector of each word of entity
        for e in entities.keys():
            w = re.findall(r"[\w']+", string.capwords(e))
            avg_vec_tmp = numpy.zeros(shape=300)
            for tmp in w:
                if tmp in model.wv.vocab:
                    avg_vec_tmp = sum(avg_vec_tmp, model.wv[tmp]) / 2
            entity_vec[e] = avg_vec_tmp

        #calculate cosine similarity
        with open(os.path.join(centroid_dir, topic, doc), 'rb') as f:
            f.seek(0)
            centroid = numpy.load(f)
        f.close()
        
        similarity_dict = {e: 1-spatial.distance.cosine(entity_vec[e], centroid) for e in entity_vec if not isnan(1-spatial.distance.cosine(entity_vec[e], centroid))}
        #similarity_dict = {e: score*(1-spatial.distance.cosine(entity_vec[e], centroid)) for e,score in entities.items() if not isnan(1-spatial.distance.cosine(entity_vec[e], centroid))}
        
        sorted_x = sorted(similarity_dict.items(), key=lambda x:x[1], reverse=True)
        if not os.path.exists(os.path.join(similarity_dir, topic)):
            os.makedirs(os.path.join(similarity_dir, topic))
        with open(os.path.join(similarity_dir, topic, doc), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sorted_x)
        f.close()
        
del model