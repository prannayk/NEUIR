import collections
import numpy as  np
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with open(filename,mode="r") as f:
    data = f.read()
    data_chars = list(set(data))
  return data.split(),data_chars,data

# Step 2: Build the dictionary and replace rare words with UNK token.


def build_dataset(words, smaller_words, vocabulary_size,dataset):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in smaller_words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

word_batch_dict = dict()
char_batch_dict = dict()
def build_everything(dataset):
  global word_batch_dict, char_batch_dict
  vocabulary_size = 100000
  with open("../data/%s/data.npy"%(dataset)) as fil:
    t = fil.readlines()
  word_max_len, char_max_len = map(lambda x: int(x),t)

  filename = '../data/corpus.txt'
  words,chars,character_data = read_data(filename)
  print('Data size', len(words))
  filename_rel = '../data/%s/corpus.txt'%(dataset)
  smaller_words, chars, character_data = read_data(filename_rel)
  char_dictionary = dict()
  for char in chars:
    char_dictionary[char] = len(char_dictionary)

  reverse_char_dictionary = dict(zip(char_dictionary.values(),char_dictionary.keys()))
  char_data = []
  for char in character_data:
    char_data.append(char_dictionary[char])

  data, count, dictionary, reverse_dictionary = build_dataset(words, smaller_words, vocabulary_size, dataset)
  del words  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

  data_index = 0
  char_data_index = 0

  word_batch_list = np.load("../data/%s/word_embedding.npy"%(dataset))
  char_batch_list = np.load("../data/%s/char_embedding.npy"%(dataset))
  print(len(word_batch_list))
  with open("../data/%s/tweet_ids.txt"%(dataset)) as fil:
    tweet_list = map(lambda x: filter(lambda y: y != '\n', x), fil.readlines())
  word_batch_dict = dict(zip(tweet_list, word_batch_list))
  char_batch_dict = dict(zip(tweet_list, char_batch_list))
  batch_list = dict()
  buffer_index = 1
  return char_batch_dict, word_batch_dict,data, count, dictionary, reverse_dictionary, word_max_len, char_max_len, vocabulary_size, char_dictionary, reverse_char_dictionary, data_index, char_data_index, buffer_index, batch_list, char_batch_list, word_batch_list, char_data

def na_loader(dataset, query):
  global word_batch_dict, char_batch_dict
  with open('../data/%s/avail.txt'%(dataset)) as f:
    tweet_list = f.readlines()
    avail_list = list()
    for tweet in tweet_list:
      tweet_name = filter(lambda x: x != '\n', tweet)
      avail_list.append([word_batch_dict[tweet_name], char_batch_dict[tweet_name]])
  with open('../data/%s/need.txt'%(dataset)) as f:
    tweet_list = f.readlines()
    need_list = list()
    for tweet in tweet_list:
      tweet_name = filter(lambda x: x != '\n', tweet)
      need_list.append([word_batch_dict[tweet_name], char_batch_dict[tweet_name]])
  return need_list, avail_list

