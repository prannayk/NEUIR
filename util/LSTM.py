import tensorflow as tf

def biLSTM_setup():
	with tf.variable_scope("lstm"):
		lstm = tf.contrib.rnn.BasicLSTMCell(embedding_size // 2, reuse=tf.get_variable_scope().reuse)
	revlstm = tf.contrib.rnn.BasicLSTMCell(embedding_size // 2, reuse=tf.get_variable_scope().reuse)
	return [lstm, revlstm]

def biLSTM_implementation(char_vector, lstm):
	batch_size = int(lstm.shape[0])
	embedding_size = int(lstm.shape[-1])
	state_fwd = lstm[0].zero_state(batch_size, dtype=tf.float32)
	state_bwd = lstm[1].zero_state(batch_size, dtype=tf.float32)
	for l in range(char_max_len):
		if l > 0:
			with tf.variable_scope("lstm"):
				cell_output_fwd, state_fwd = lstm[0](character_word_embeddings[:,l], state_fwd)
			cell_output_bwd, state_bwd = lstm[1](character_word_embeddings[:, char_max_len - l - 1], state_fwd)
			cell_output_fwd = tf.reshape(cell_output_fwd, shape=[batch_size,1,embedding_size // 2])
			cell_output_bwd = tf.reshape(cell_output_bwd, shape=[batch_size,1,embedding_size // 2])
		else:
			with tf.variable_scope("lstm"):
		        cell_output_fwd, state_fwd = lstm[0](character_word_embeddings[:,l],state_fwd)
		    cell_output_bwd, state_bwd = lstm[1](character_word_embeddings[:,l],state_bwd)
		    output_fwd = tf.reshape(cell_output_fwd, shape=[batch_size,1,embedding_size//2])
		    output_bwd = tf.reshape(cell_output_bwd, shape=[batch_size,1,embedding_size//2])
	intermediate = tf.concat([output_fwd, output_bwd], axis=2)
	return intermediate

def attention(w1, w2, attention_input):
	weights = tf.stack([w1]*batch_size)
	vvector = tf.stack([w2]*batch_size)
	attention = tf.nn.softmax(tf.matmul(vvector, tf.nn.tanh(tf.matmul(attention_input,weights)),transpose_a=True))
	output = tf.reshape(tf.matmul(attention,intermediate),shape=[batch_size,embedding_size])
	return output