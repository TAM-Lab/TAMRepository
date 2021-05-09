import tensorflow as tf


# inputs: (?, max_sen_length, hidden_size)
def attention(inputs):
    # Trainable parameters
    hidden_size = inputs.shape[2].value
    u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

    with tf.name_scope('v'):
        v = tf.tanh(inputs)  # M = tanh(H)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape  alpha = softmax(wT * M)
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1) # r = H * alpha T

    # Final output with tanh
    output = tf.tanh(output)

    return output, alphas

