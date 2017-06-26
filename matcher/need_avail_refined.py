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
from setup import *

dataset, query_type, filename, num_steps, num_steps_roll, num_steps_train, expand_flag,lr_, matchname, counter = import_arguments(sys.argv)

# Read the data into a list of strings.
# import data
char_batch_dict, word_batch_dict,data, count, dictionary, reverse_dictionary, word_max_len, char_max_len, vocabulary_size, char_dictionary, reverse_char_dictionary, data_index, char_data_index, buffer_index, batch_list, char_batch_list, word_batch_list, char_data = build_everything(dataset)


data_index, batch, labels = generate_batch(data, data_index, batch_size=8, num_skips=2, skip_window=1,)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
char_data_index, batch, labels = generate_batch_char(char_data, char_data_index, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_char_dictionary[batch[i]],
        '->', labels[i, 0], reverse_char_dictionary[labels[i, 0]])

lambda_1, tweet_batch_size, expand_start_count, query_name, query_tokens, query_tokens_alternate, char_batch_size, num_sampled, valid_examples, valid_window, valid_size, skip_window, num_skips, embedding_size, char_vocabulary_size, batch_size, num_char_skips, skip_char_window = setup(char_dictionary, dictionary, query_type)

learning_rate = lr_

expand_count = 3
need_tweet_list , avail_tweet_list = na_loader(dataset, query_name)
saver = tf.train.import_meta_graph('../results/%s/%s/%s_model.ckpt.meta'%(dataset, query_name, filename))
print("Loaded graph")
graph = tf.get_default_graph()

expand_count = 3
with tf.Session(graph=graph) as session:
  count = 0
  saver.restore(session, '../results/%s/%s/%s_model.ckpt'%(dataset, query_name, filename))
  query_tweet_holder = [graph.get_tensor_by_name('tweet_query_word_holder:0'), graph.get_tensor_by_name('tweet_query_char_holder:0')]
  tweet_similarity = graph.get_tensor_by_name('match_similarity:0')
  tweet_word_holder = graph.get_tensor_by_name('tweet_word_holder:0')
  tweet_char_holder = graph.get_tensor_by_name('tweet_char_holder:0')
  for query_tweet in need_tweet_list:
    count = print_tweets(dataset, tweet_similarity, query_tweet, query_tweet_holder, query_name, session, avail_tweet_list, char_batch_list, tweet_word_holder, tweet_char_holder, count ,tweet_batch_size, "%s_%s"%(filename, matchname), True, counter)
    if count % 25 == 0: print("Completed for %d need tweets, saved as avail list"%(count))

