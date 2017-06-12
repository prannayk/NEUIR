import time
def run_iteration(generator, placeholders, loss, optimizer, session, filename, batch_size, num_skips, skip_window, data, data_index):
    data_index, batch, label = generator(data, data_index, batch_size, num_skips, skip_window)
    feed_dict = dict(zip(placeholders, [batch, label]))
    _,loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    return loss_val, data_index

def train_model(session, dataset,query_similarity, query_name, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, generators, similarities, num_steps, placeholders,loss, optimizers, interval1, interval2, valid_size, reverse_dictionary, batch_size, num_skips, skip_window, filename,data, data_index):
    average_loss = list()
    g_len = len(generators)
    for t in range(g_len):
        average_loss.append(0)
    for step in xrange(num_steps):
        for i in range(g_len):
            loss_val,data_index[i] = run_iteration(generators[i],placeholders[i], loss[i], optimizers[i], session, filename, batch_size, num_skips, skip_window,data[i], data_index[i])
            average_loss[i] += loss_val
        if step % interval1 == 0 and step > 0:
            for t in range(g_len):
                start_time[i], count[i] = standard_print_fn(filename, step, average_loss[t], start[t], interval1, count[t])
        elif step == 0:
            start_time = []
            count = []
            for _ in range(g_len):
                start_time.append(time.time())
                count.append(0)
        if step % interval2 == 0:
            for t in range(len(similarities)):
                sim = similarities[t].eval()
                rank_tokens(valid_size[i], reverse_dictionary[i], 8,sim)
            print_tweets(dataset, query_type, session, word_batch_list, char_batch_list, tweet_word_holder, tweet_char_holder, count)

