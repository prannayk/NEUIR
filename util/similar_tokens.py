def rank_tokens(valid_size,valid_examples, reverse_dictionary, top_k,sim):
	for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        word_list = []
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          word_list.append(close_word)
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
       	return word_list