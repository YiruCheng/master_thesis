# Weakly Supervised Topic Prediction in Political Documents
This is an introduction for my master thesis "Weakly Supervised Topic Prediction in Political Documents", supervised by Dr. Federico Nanni in University of Mannheim.

Note: Here is not the entire code for the work.

Description
------------
a) Document-based: scripts of baseline, frequency, TF-IDF and PageRank

b) Wikipedia-based: scripts of baseline, frequency and TF-IDF

c) Entity_Embedding: scripts of evaluation methods (frequency and TF-IDF) and calculating centroid and similarity based on different ranking methods

  * centroid needs to be process firstly, and then similarity.
  * model comes from Google https://code.google.com/archive/p/word2vec/

d) Word_Embedding: same as Entity_Embedding

e)traditional_Chinese: same as Entity_Embedding
	
  * pre-trained_vectors contains the embedding from Wikipedia https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

f) evaluation: scripts of our evaluation methods (MAP and Recall@k) 

g) pywikibot: api bot for querying Wikipedia
	
  * needs to login (check README.rst for the progress)
	
  * scripts-wiki_pages.py: collecting all edges based on Wikipedia-based dataset
	
  * scripts-doc_pages.py: collecting all edges based on Document-based dataset

Execution Notes
----------------
1) Python scripts are using Python 3.6
