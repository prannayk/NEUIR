# import json
# import python
import os
import system
from print_tweets import *

def expand_query(dataset, similarity, word_batch_dict, top_k):
	top_tweets = top_tweets(top_k)	
	word_batch_list = []
	for tweet in tweet_list:
		word_batch_list += word_batch_dict[tweet]
	sim = similarity.eval()
	nearest = filter(lambda x: x in word_batch_list,-sim.argsort())
	print(nearest)
	return nearest
