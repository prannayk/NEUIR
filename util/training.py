def run_iteration(generator, placeholders, loss, optimizer, session, filename, batch_size, num_skips, skip_window)
	feed_dict = dict(zip(placeholders, generator(batch_size, num_skips, skip_window)))
	_,loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
	return average_loss + loss_val

def train_model(session, dataset,query_similarity, query_name, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, generators, similarities, num_steps, placeholders,loss, optimizers, interval1, interval2, valid_size, reverse_dictionary, batch_size, num_skips, skip_window):
	average_losses = list()
	g_len = len(generators)
	for t in range(g_len):
		average_losses.append(0)
	for step in xrange(num_steps):
		for i in range(g_len):
			average_losses[i] = run_iteration(generators[i],placeholders[i], loss[i], optimizers[i], session)
		if step % interval1 == 0 and step > 0:
			for t in range(g_len):
				start_time[i], count[i] = standard_print_fn(filename, step, average_loss[t], start[t], interval1, count[t])
		elif step == 0:
			start_time = []
			count = []
			for t in g_len:
				start_time.append(time.time())
				count[0]
		if step % interval2 == 0:
			for t in range(len(similarities)):
				sim = similarities[t].eval()
				rank_tokens(valid_size[i], reverse_dictionary[i], top_k=8,sim)
			print_tweets(dataset, query_type, session, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, count)

