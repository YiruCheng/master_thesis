# -*- coding: utf-8 -*-
"""
@author: Yi-Ru Cheng
"""

import numpy
from scipy import spatial
import gensim
import os
import csv
from collections import OrderedDict


dataset_dir = '../../chinese-dataset/CKIP'
centroid_dir = 'EE_freq200_centroid'
similarity_dir = 'EE_freq200_similarity'

model = gensim.models.KeyedVectors.load_word2vec_format('pre-trained_vectors/wiki.zh_classical.vec')
for topic in os.listdir(dataset_dir):
    print('***********************'+topic)
    topic_dir = os.path.join(dataset_dir, topic)
    
    if not os.path.exists(os.path.join(similarity_dir, topic)):
        os.makedirs(os.path.join(similarity_dir, topic))
    
    for doc in os.listdir(topic_dir):
        with open(os.path.join(topic_dir, doc), 'r', encoding='utf-8-sig') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        f.close()
        
        entities = {}
        for e in content:
            if e in entities.keys():
                curr_freq = entities[e]
                entities[e] = curr_freq+1
            else:
                entities[e] = 1
        
        entities = OrderedDict(sorted(entities.items(), key=lambda x: x[1], reverse=True)[:200])

        #calculate consine similarity
        with open(os.path.join(centroid_dir, topic, doc), 'rb') as f:
            f.seek(0)
            centroid = numpy.load(f)
        f.close()
        
        similarity_dict = {e: 1-spatial.distance.cosine(model[e], centroid) for e in entities.keys() if e in model}
        
        sorted_x = sorted(similarity_dict.items(), key=lambda x:x[1], reverse=True)
        with open(os.path.join(similarity_dir, topic, doc), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(sorted_x)
        f.close()
        
del model