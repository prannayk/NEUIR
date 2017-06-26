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

dataset, query_type, filename, num_steps, num_steps_roll, num_steps_train, expand_flag,lr_, matchname = import_arguments(sys.argv)
char_batch_dict, word_batch_dict,data, count, dictionary, reverse_dictionary, word_max_len, char_max_len, vocabulary_size, char_dictionary, reverse_char_dictionary, data_index, char_data_index, buffer_index, batch_list, char_batch_list, word_batch_list, char_data = build_everything(dataset)

initialze_batch_values(8,data, char_data, [], 1, 1, 2, 2)

data_index, batch, labels = generate_batch()
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
char_data_index, batch, labels = generate_batch_char()
for i in range(8):
  print(batch[i], reverse_char_dictionary[batch[i]],
        '->', labels[i, 0], reverse_char_dictionary[labels[i, 0]])

lambda_1, tweet_batch_size, expand_start_count, query_name, query_tokens, query_tokens_alternate, char_batch_size, num_sampled, valid_examples, valid_window, valid_size, skip_window, num_skips, embedding_size, char_vocabulary_size, batch_size, num_char_skips, skip_char_window = setup(char_dictionary, dictionary, query_type)
learning_rate = lr_
initialze_batch_values(batch_size,data, char_data, [], 1, 1, 2, 2)

with graph.as_default():

  # Input data.
  need_constant = tf.constant(query_tokens,dtype=tf.int32)
  avail_constant = tf.constant(query_tokens_alternate, dtype=tf.int32)
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_input_chars = tf.placeholder(tf.int32, shape=[char_batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  train_char_labels = tf.placeholder(tf.int32, shape=[char_batch_size, 1])
  valid_dataset = tf.constant(valid_examples[0], dtype=tf.int32)
  word_char_embeddings = tf.placeholder(tf.int32, shape=[batch_size, char_max_len])
  valid_char_dataset = tf.constant(valid_examples[1], dtype=tf.int32)
  expanded_query_ints = tf.placeholder(tf.int32, shape=(len(query_tokens)+3))
  query_ints = tf.placeholder(tf.int32, shape=len(query_tokens))
  tquery_word_holder = tf.placeholder(tf.int32, shape=[word_max_len],name="tweet_query_word_holder")
  tquery_char_holder = tf.placeholder(tf.int32, shape=[word_max_len, char_max_len],name="tweet_query_char_holder")
  # Ops and variables pinned to the CPU because of missing GPU implementation
  tweet_char_holder = tf.placeholder(tf.int32, shape=[tweet_batch_size,word_max_len,char_max_len],name="tweet_char_holder")
  tweet_word_holder = tf.placeholder(tf.int32, shape=[tweet_batch_size, word_max_len],name="tweet_word_holder")
    # Look up embeddings for inputs.
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name="embeddings")
  char_embeddings = tf.Variable(tf.random_uniform([char_vocabulary_size, embedding_size],-1.0,1.0),name="char_embeddings")
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)
  char_embed = tf.nn.embedding_lookup(char_embeddings,train_input_chars)

  # Construct the variables for the NCE loss
  nce_weights = tf.Variable(
      tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)),name="nce_weights")
  nce_biases = tf.Variable(tf.zeros([vocabulary_size]),name="nce_biases")
  # character weights
  nce_char_weights = tf.Variable(
      tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)),name="nce_char_weights")
  nce_char_biases = tf.Variable(tf.zeros([vocabulary_size]),name="nce_char_biases")

    
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  loss_char = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_char_weights,
                     biases=nce_char_biases,
                     labels=train_char_labels,
                     inputs=char_embed,
                     num_sampled=10,
                     num_classes=char_vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  optimizer_char = tf.train.AdamOptimizer(learning_rate /5).minimize(loss_char)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
  query_embedding_token = tf.reshape(tf.reduce_mean(tf.nn.embedding_lookup(normalized_embeddings,query_ints),axis=0),shape=[1,embedding_size])
  expanded_query_embedding_token = tf.reshape(tf.reduce_mean(tf.nn.embedding_lookup(normalized_embeddings,expanded_query_ints),axis=0),shape=[1,embedding_size])
  similarity_query = tf.reshape(tf.matmul(
      query_embedding_token, normalized_embeddings, transpose_b=True),shape=[int(normalized_embeddings.shape[0])])
  similarity_expanded_query = tf.reshape(tf.matmul(
      expanded_query_embedding_token, normalized_embeddings, transpose_b=True),shape=[int(normalized_embeddings.shape[0])])

  norm_char = tf.sqrt(tf.reduce_sum(tf.square(char_embeddings), 1, keep_dims=True))
  normalized_char_embeddings = char_embeddings / norm_char
  valid_embeddings_char = tf.nn.embedding_lookup(
      normalized_char_embeddings, valid_char_dataset)
  similarity_char = tf.matmul(
      valid_embeddings_char, normalized_char_embeddings, transpose_b=True)

  tweet_word_embed = tf.nn.embedding_lookup(normalized_embeddings, tweet_word_holder)
  tweet_char_embed = tf.reduce_mean(tf.nn.embedding_lookup(normalized_char_embeddings, tweet_char_holder),axis=2)
  tweet_embedding = tf.reduce_mean(lambda_1*tweet_word_embed + (1-lambda_1)*tweet_char_embed,axis=1)
  query_embedding = tf.reshape(tf.reduce_mean(tf.nn.embedding_lookup(normalized_embeddings,query_ints),axis=0),shape=[1,embedding_size])
  expanded_query_embedding = tf.reshape(tf.reduce_mean(tf.nn.embedding_lookup(normalized_embeddings,expanded_query_ints),axis=0),shape=[1,embedding_size])
  query_similarity = tf.reshape(tf.matmul(tweet_embedding, query_embedding, transpose_b=True),shape=[tweet_batch_size])
  expanded_query_similarity = tf.reshape(tf.matmul(tweet_embedding, expanded_query_embedding, transpose_b=True),shape=[tweet_batch_size])
  
  tweet_query_char = tf.reduce_mean(tf.nn.embedding_lookup(normalized_char_embeddings, tquery_char_holder),axis=1)
  tweet_query_word = tf.nn.embedding_lookup(normalized_embeddings, tquery_word_holder)
  tquery_embedding = tf.reshape(tf.reduce_mean(lambda_1*tweet_query_word + lambda_1*tweet_query_char,axis=0),shape=[1,embedding_size])
  
  norm_query = tf.sqrt(tf.reduce_sum(tf.square(tquery_embedding), 1, keep_dims=True))
  tquery_embedding_norm = tquery_embedding / norm_query
  cosine = tf.matmul(tweet_embedding, tquery_embedding_norm, transpose_b=True)
  tweet_query_similarity = tf.reshape(cosine, shape=[tweet_batch_size], name="tweet_query_similarity")
 
  tquery_embedding_norm_dim = tf.reshape(tquery_embedding_norm, shape=[1,embedding_size])
  query_need_embedding = tf.reshape(tf.reduce_mean(tf.nn.embedding_lookup(normalized_embeddings, need_constant),axis=0),shape=[1,embedding_size])
  cosine_need = tf.matmul(tquery_embedding_norm_dim, query_need_embedding, transpose_b=True)
  tquery_embedding_reqd = tf.reshape(tquery_embedding_norm_dim - (cosine_need*tquery_embedding_norm_dim),shape=[1,embedding_size])
  # we have the need vector without the need vector
  query_avail_embedding = tf.reshape(tf.reduce_mean(tf.nn.embedding_lookup(normalized_embeddings,avail_constant),axis=0),shape=[1,embedding_size])
  query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_avail_embedding),1,keep_dims=True))
  query_avail_embedding_norm = query_embedding / query_norm
  cosine_avail = tf.matmul(tweet_embedding, query_avail_embedding_norm, transpose_b=True)
  reduced_tweet_embedding = tweet_embedding - (tweet_embedding*cosine_avail)
  match_similarity = tf.reshape(tf.matmul(reduced_tweet_embedding, tquery_embedding_reqd, transpose_b=True),shape=[tweet_batch_size],name="match_similarity")


  # Add variable initializer logger

  embeddings_copy = tf.Variable(tf.zeros_like(embeddings),name="embeddings_copy")
  char_embeddings_copy = tf.Variable(tf.zeros_like(char_embeddings),name="char_embeddings_copy")

  # Construct the variables for the NCE loss
  nce_weights_copy = tf.Variable(tf.zeros_like(nce_weights),name="nce_weights_copy")
  nce_biases_copy = tf.Variable(tf.zeros([vocabulary_size]),name="nce_biases_copy")
  # character weights
  nce_char_weights_copy = tf.Variable(tf.zeros_like(nce_char_weights),name="nce_char_weights_copy")
  nce_char_biases_copy = tf.Variable(tf.zeros([vocabulary_size]),name="nce_char_biases_copy")

  assign = [embeddings_copy.assign(embeddings), char_embeddings_copy.assign(char_embeddings),
    nce_weights_copy.assign(nce_weights), nce_biases_copy.assign(nce_biases),
    nce_char_weights_copy.assign(nce_char_weights), nce_char_biases_copy.assign(nce_char_biases)]


