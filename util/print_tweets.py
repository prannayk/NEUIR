import time
import numpy as np
sorted_tweets = []
tweet_count = 0
tweet_list = []
def print_tweets(dataset, query_similarity, query_tokens, query_token_holder, query_name, session, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, count, tweet_batch_size, filename, flag=False):
  global sorted_tweets, tweet_count
  if tweet_list == []:
    load_tweet(dataset)
  folder_name = '../results/%s/%s/%s/'%(dataset, query_name,filename)
  tweet_embedding_val = []
  if flag :
    word_batch = list(word_batch_list)
    word_batch_list = []
    char_batch_list = []
    for t in word_batch:
      word_batch_list.append(t[0])
      char_batch_list.append(t[1])
  for t in range(len(word_batch_list) // tweet_batch_size):
    if not flag:
      feed_dict = {
        tweet_word_holder : word_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size],
        tweet_char_holder : char_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size],
        query_token_holder : np.array(query_tokens)
      }
    else:
      feed_dict = {
        tweet_word_holder : word_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size],
        tweet_char_holder : char_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size],
        query_token_holder[0] : np.array(query_tokens[0]),
        query_token_holder[1] : np.array(query_tokens[1])
      }
    l = session.run(query_similarity, feed_dict = feed_dict)
    if len(tweet_embedding_val) % 25 == 0 :
      if not flag : 
        print(len(tweet_embedding_val))
    tweet_embedding_val += list(l) 
  tweet_embedding_dict = dict(zip(tweet_list, tweet_embedding_val))
  sorted_tweets = [i for i in sorted(tweet_embedding_dict.items(), key=lambda x: -x[1])]
  count += 1
  file_list = []
  for i in range(len(sorted_tweets)):
    dataset_name = list(dataset)
    dataset_name[0] = dataset[0].upper()
    dataset_name[1:] = dataset[1:]
    file_list.append('%s-%s 0 %s %d %f running'%(dataset, query_name,sorted_tweets[i][0],i+1,sorted_tweets[i][1]))
  with open("%stweet_list_%d.txt"%(folder_name,count),mode="w") as fw:
    fw.write('\n'.join(map(lambda x: str(x),file_list)))
  return count

def standard_print_fn(filename, step, average_loss, start, density, count):
  print("Running %s at %d where the average_loss is : %f"%(filename, step, average_loss/density))
  print("Time taken for said iteration was: %f.2"%(time.time()-start))
  return time.time(), count+1

def load_tweet(dataset):
  global tweet_list
  with open("../data/%s/tweet_ids.txt"%(dataset)) as fil:
    tweet_list = fil.readlines()
    tweet_list = map(lambda y: filter(lambda x: x != '\n', y), tweet_list)

def initialize(tweet_c):
  global tweet_count
  tweet_count = tweet_c

def top_tweets(top_k):
  global sorted_tweets
  print(len(sorted_tweets))
  print("Returning top tweets' ids:")
  top_tweets = map(lambda x: x[0],sorted_tweets[:top_k])
  if sorted_tweets != []:
    for i in range(10):
      print(top_tweets[0])
  return top_tweets
