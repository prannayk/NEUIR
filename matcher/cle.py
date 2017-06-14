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
word_batch_dict,data, count, dictionary, reverse_dictionary, word_max_len, char_max_len, vocabulary_size, char_dictionary, reverse_char_dictionary, data_index, char_data_index, buffer_index, batch_list, char_batch_list, word_batch_list, char_data = build_everything(dataset)
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

batch_size = 256
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
char_batch_size = 256
if query_type == 0 :
  query_tokens = map(lambda x: dictionary[x],['nee','requir'])
else :
  query_tokens = map(lambda x: dictionary[x],['send','distribut','avail'])
tweet_batch_size = 256
lambda_1 = 0.7

saver = tf.import_meta_graph()
graph = tf.get_default_graph()
learning_rate = 5e-1



# loading tweet list in integer marking form
# load more data
expand_count = 3
with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  trainable = [i for i in filter(lambda x: x.startswith(filename),tf.trainable_variables())]
  saver.load()
  init.run()
  count = 0
  print("Initialized")

  generators = [generate_batch, generate_batch_char]
  similarities = [similarity, similarity_char]
  placeholders = [[train_inputs,train_labels],[train_input_chars,train_char_labels]]
  losses = [loss, loss_char]
  optimizers = [optimizer, optimizer_char]
  interval1 = 2000
  interval2 = 10000
  datas = [data,char_data]
  data_index = [data_index, char_data_index, buffer_index]
  reverse_dictionaries = [reverse_dictionary, reverse_char_dictionary]
  if query_type == 0:
    query_name = 'Need'
  else :
    query_name == 'Avail'
  print(query_tokens)
  train_model(session, dataset,query_similarity, query_tokens, query_ints, query_name, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, generators, similarities, num_steps, placeholders,losses, optimizers, interval1, interval2, valid_size, valid_examples, reverse_dictionaries, batch_size, num_skips, skip_window, filename , datas, data_index, tweet_batch_size)
  placeholders += [[train_inputs, word_char_embeddings, train_labels]]
  train_model(session, dataset,query_similarity, query_tokens ,query_ints, query_name, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, generators, similarities, num_steps_roll, placeholders,losses, optimizers, interval1, interval2, valid_size, valid_examples, reverse_dictionaries, batch_size, num_skips, skip_window, filename, datas, data_index, tweet_batch_size)
  
  expanded_query_tokens, expanded_query_holder, final_query_similarity= expand_query(expand_flag, session,query_ints, np.array(query_tokens),dataset ,similarity_query, word_batch_dict, 100, query_ints, expanded_query_ints, query_similarity, expanded_query_similarity)
  expanded_query_tokens = query_tokens + expanded_query_tokens[2:2+expand_count]
  print(expanded_query_tokens)
  
  train_model(session, dataset,final_query_similarity, expanded_query_tokens, expanded_query_holder, query_name, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, generators, similarities, num_steps_train , placeholders,losses, optimizers, interval1, interval2, valid_size, valid_examples, reverse_dictionaries, batch_size, num_skips, skip_window, filename , datas, data_index, tweet_batch_size)
  folder_name = './%s/%s/'%(dataset, query_type)
  final_embeddings = normalized_embeddings.eval()
  final_char_embedding = normalized_char_embeddings.eval()
  np.save('%sword_embeddings.npy', final_embeddings)
  np.save('%schar_embeddings.npy', final_char_embedding)
