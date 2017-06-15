import sys
import numpy as np
sys.path.append( '../util/')
from generators import *
from loader import *
from print_tweets import *
from similar_tokens import * 
from training import *
from similar_tokens import *
from expand_query import *
from argument_loader import *
	
def setup(char_dictionary, dictionary, query_type):	
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 2       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.
    skip_char_window = 2
    num_char_skips = 3
    char_vocabulary_size = len(char_dictionary)
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
        expand_start_count = 2
    else :
        query_tokens = map(lambda x: dictionary[x],['send','distribut','avail'])
        query_name = "Avail"
        expand_start_count = 3
    tweet_batch_size = 128
    lambda_1 = 0.7
    return lambda_1, tweet_batch_size, expand_start_count, query_name, query_tokens, char_batch_size, num_sampled, valid_examples, valid_window, valid_size, skip_window, num_skips, embedding_size, char_vocabulary_size, batch_size, num_char_skips, skip_char_window
