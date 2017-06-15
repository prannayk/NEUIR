from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import operator
import collections
import math
import time
import os
import random
import zipfile
import time
import numpy as np
import sys
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
sys.path.append( '../util/')
from generators import *
from loader import *
from print_tweets import *
from similar_tokens import * 
from training import *
from similar_tokens import *
from expand_query import *
from argument_loader import *

dataset, query_type, filename, num_steps, num_steps_roll, num_steps_train, expand_flag = import_arguments(sys.argv)

# Read the data into a list of strings.
# import data
char_batch_dict, word_batch_dict,data, count, dictionary, reverse_dictionary, word_max_len, char_max_len, vocabulary_size, char_dictionary, reverse_char_dictionary, data_index, char_data_index, buffer_index, batch_list, char_batch_list, word_batch_list, char_data = build_everything(dataset)
# Step 3: Function to generate a training batch for the skip-gram model.


data_index, batch, labels = generate_batch(data, data_index, batch_size=8, num_skips=2, skip_window=1,)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
char_data_index, batch, labels = generate_batch_char(char_data, char_data_index, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_char_dictionary[batch[i]],
        '->', labels[i, 0], reverse_char_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
skip_char_window = 2
num_char_skips = 3
char_vocabulary_size = len(char_dictionary)
print(char_vocabulary_size)
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = list()
valid_window = list()
valid_size.append(16)     # Random set of words to evaluate similarity on.
valid_size.append(10)
valid_window.append(100)  # Only pick dev samples in the head of the distribution.
valid_window.append(20)
valid_examples = []
valid_examples.append(np.random.choice(valid_window[0], valid_size[0], replace=False))
valid_examples.append(np.random.choice(valid_window[1], valid_size[1], replace=False))
valid_examples[0][0] = dictionary['nee']
valid_examples[0][1] = dictionary['avail']
num_sampled = 64    # Number of negative examples to sample.
char_batch_size = 128
if query_type == 0 :
  query_tokens = map(lambda x: dictionary[x],['nee','requir'])
  query_name = "Need"
else :
  query_tokens = map(lambda x: dictionary[x],['send','distribut','avail'])
  query_name = "Avail"
tweet_batch_size = 128
lambda_1 = 0.7

learning_rate = 5e-1

expand_count = 3
need_tweet_list , avail_tweet_list = na_loader(dataset, query_name)
saver = tf.train.import_meta_graph('../results/%s/%s/%s_model.ckpt.meta'%(dataset, query_name, filename))
print("Loaded graph")
graph = tf.get_default_graph()

expand_count = 3

with tf.Session(graph=graph) as session:
  print("inside session")
  count = 0
  saver.restore(session, '../results/%s/%s/%s_model.ckpt'%(dataset, query_name, filename))
  query_tweet_holder = [graph.get_tensor_by_name('tweet_query_word_holder:0'), graph.get_tensor_by_name('tweet_query_char_holder:0')]
  tweet_similarity = graph.get_tensor_by_name('tweet_query_similarity:0')
  tweet_word_holder = graph.get_tensor_by_name('tweet_word_holder:0')
  tweet_char_holder = graph.get_tensor_by_name('tweet_char_holder:0')
  for query_tweet in need_tweet_list:
    count = print_tweets(dataset, tweet_similarity, query_tweet, query_tweet_holder, query_name, session, avail_tweet_list, char_batch_list, tweet_word_holder, tweet_char_holder, count ,tweet_batch_size, "%s_match_data"%(filename), True)
    if count % 100 == 0: print("Completed for %d need tweets, saved as avail list"%(count))

