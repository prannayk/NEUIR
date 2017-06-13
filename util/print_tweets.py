import time

tweet_list = []

def print_tweets(dataset, query_similarity, query_name, session, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, count, tweet_batch_size, filename):
  global tweet_list
  if tweet_list == []:
    load_tweet(dataset)
  folder_name = '../results/%s/%s/%s/'%(dataset, query_name,filename)
  tweet_embedding_val = []
  for t in range(len(word_batch_list) // tweet_batch_size):
    feed_dict = {
      tweet_word_holder : word_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size],
      tweet_char_holder : char_batch_list[t*tweet_batch_size:t*tweet_batch_size + tweet_batch_size]
    }
    l = session.run(query_similarity, feed_dict = feed_dict)
    if len(tweet_embedding_val) % 1000 == 0 :
      print(len(tweet_embedding_val))
    tweet_embedding_val += list(l) 
  tweet_embedding_dict = dict(zip(tweet_list, tweet_embedding_val))
  sorted_tweets = [i for i in sorted(tweet_embedding_dict.items(), key=lambda x: -x[1])]
  count += 1
  file_list = []
  for i in range(len(sorted_tweets)):
    file_list.append('%s-%s 0 %s %d %f running'%(dataset, query_name,sorted_tweets[i][0],i+1,sorted_tweets[i][1]))
  with open("%stweet_list_%d.txt"%(folder_name,count),mode="w") as fw:
    fw.write('\n'.join(map(lambda x: str(x),file_list)))
  return count

def standard_print_fn(filename, step, average_loss, start, density, count):
  print("Running %s at %d where the average_loss is : %f"%(filename, step, average_loss/density))
  print("Time taken for said iteration was: ",(time.time()-start))
  return time.time(), count+1

def load_tweet(dataset):
  global tweet_list
  with open("../data/%s/tweet_ids.txt"%(dataset)) as fil:
    tweet_list = fil.readlines()
    tweet_list = map(lambda y: filter(lambda x: x != '\n', y), tweet_list)
