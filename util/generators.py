import numpy as np
import collections
import random
def generate_batch(data, data_index, batch_size, num_skips, skip_window):
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return data_index, batch, labels

def generate_batch_char(char_data, char_data_index, batch_size, num_skips, skip_window):
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(char_data[char_data_index])
    char_data_index = (char_data_index + 1) % len(char_data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(char_data[char_data_index])
    char_data_index = (char_data_index + 1) % len(char_data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  char_data_index = (char_data_index + len(char_data) - span) % len(char_data)
  return char_data_index, batch, labels

def generate_batch_train(data, buffer_index, batch_size, num_skips, skip_window):
  global buffer_index
  train_data_index = 0
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  batch_chars = np.ndarray(shape=(batch_size, char_max_len),dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  l = batch_size // word_max_len
  word_data = np.ndarray(shape=[l*word_max_len])
  char_data = np.ndarray(shape=[l*word_max_len,char_max_len])
  for i in range(l):
   word_data[word_max_len*i:word_max_len*(i+1)] = word_batch_list[buffer_index]
   char_data[word_max_len*i:word_max_len*(i+1)] = char_batch_list[buffer_index]
   buffer_index = (buffer_index + 1) % len(word_batch_list)
  buffer = collections.deque(maxlen=span)
  buffer_ = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(word_data[train_data_index])
    buffer_.append(char_data[train_data_index])
    train_data_index = (train_data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      batch_chars[i*num_skips + j] = buffer_[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(word_data[train_data_index])
    buffer_.append(char_data[train_data_index])
    train_data_index = (train_data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  train_data_index = (train_data_index + len(word_data) - span) % len(word_data)
  return buffer_index, batch, batch_chars, labels
