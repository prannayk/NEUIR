import time
from print_tweets import *
def rank_tokens(valid_examples,valid_size, reverse_dictionary, top_k,sim):
    word_list = dict()
    for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        word_list[valid_word] = []
        for k in xrange(top_k):
            close_word = reverse_dictionary[nearest[k]]
            word_list[valid_word].append(close_word)
            log_str = "%s %s," % (log_str, close_word)
        print(log_str)
    return word_list
def run_iteration(generator, placeholders, loss, optimizer, session, filename, batch_size, num_skips, skip_window, data, data_index):
    batch_data = generator(data, data_index, batch_size, num_skips, skip_window)
    data_index = batch_data[0]
    feed_dict = dict(zip(placeholders, batch_data[1:]))
    _,loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    return loss_val, data_index

# main training model
def train_model(session, dataset,query_similarity, query_tokens, query_token_holder, query_name, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, generators, similarities, num_steps, placeholders,loss, optimizers, interval1, interval2, valid_size, valid_examples, reverse_dictionary, batch_size, num_skips, skip_window, filename,data, data_index, tweet_batch_size):
    average_loss = list()
    count_ = 0
    g_len = len(generators)
    for t in range(g_len):
        average_loss.append(0)
    for step in xrange(num_steps):
        for i in range(g_len):
            loss_val,data_index[i] = run_iteration(generators[i],placeholders[i], loss[i], optimizers[i], session, filename, batch_size, num_skips, skip_window,data[i], data_index[i])
            average_loss[i] += loss_val
        if step % interval1 == 0 and step > 0:
            for t in range(g_len):
                start_time[t], count[t] = standard_print_fn(filename, step, average_loss[t], start_time[t], interval1, count[t])
                average_loss[t] = 0
        elif step == 0:
            start_time = []
            count = []
            for _ in range(g_len):
                start_time.append(time.time())
                count.append(0)
        if step % interval2 == 0:
            for t in range(len(similarities)):
                sim = similarities[t].eval()
                rank_tokens(valid_examples[t],valid_size[t], reverse_dictionary[t], 8,sim)
            count_ = print_tweets(dataset, query_similarity, query_tokens, query_token_holder, query_name, session, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, count_, tweet_batch_size, filename)

