import tensorflow as tf

def biLSTM_setup(embedding_size):
	with tf.variable_scope("lstm"):
		lstm = tf.contrib.rnn.BasicLSTMCell(embedding_size // 2, reuse=tf.get_variable_scope().reuse)
	revlstm = tf.contrib.rnn.BasicLSTMCell(embedding_size // 2, reuse=tf.get_variable_scope().reuse)
	return [lstm, revlstm]

def biLSTM_implementation(char_vector, lstm, flag=True):
    batch_size = int(char_vector.shape[0])
    embedding_size = 2*int(char_vector.shape[-1])
    char_max_len = int(char_vector.shape[-2])
    print("Character MaxLlen : %d and embedding_size: %d"%(char_max_len, embedding_size))
    state_fwd = lstm[0].zero_state(batch_size, dtype=tf.float32)
    state_bwd = lstm[1].zero_state(batch_size, dtype=tf.float32)
    repeat_flag = False
    for l in range(char_max_len):
        if flag or repeat_flag :
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.variable_scope("lstm"):
                    cell_output_fwd, state_fwd = lstm[0](char_vector[:,l], state_fwd)
                cell_output_bwd, state_bwd = lstm[1](char_vector[:, char_max_len - l - 1], state_bwd)
        else:
            repeat_flag = True
            with tf.variable_scope("lstm"):
                cell_output_fwd, state_fwd = lstm[0](char_vector[:,l], state_fwd)
            cell_output_bwd, state_bwd = lstm[1](char_vector[:,char_max_len - l - 1], state_bwd)
        rcell_output_fwd = tf.reshape(cell_output_fwd, shape=[batch_size,1,embedding_size // 2])
        rcell_output_bwd = tf.reshape(cell_output_bwd, shape=[batch_size,1,embedding_size // 2])
        if l == 0:
            output_fwd = rcell_output_fwd
            output_bwd = rcell_output_bwd
        else:
            output_fwd = tf.concat([output_fwd, rcell_output_fwd],axis=1)
            output_bwd = tf.concat([rcell_output_bwd, output_bwd],axis=1)
    intermediate = tf.concat([output_fwd, output_bwd], axis=2)
    return intermediate

def attention(w1, w2, attention_input):
    batch_size = int(attention_input.shape[0])
    embedding_size = int(attention_input.shape[-1])
    weights = tf.stack([w1]*batch_size)
    vvector = tf.stack([w2]*batch_size)
    attention = tf.nn.softmax(tf.matmul(vvector, tf.nn.tanh(tf.matmul(attention_input,weights)),transpose_a=True,transpose_b=True))
    output = tf.reshape(tf.matmul(attention,attention_input),shape=[batch_size,embedding_size])
    return output
