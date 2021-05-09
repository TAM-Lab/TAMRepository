import tensorflow as tf


def attention(inputs):
    # Trainable parameters
    hidden_size = inputs.shape[2].value
    u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

    v = inputs

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')
    alphas = tf.nn.softmax(vu, name='alphas')
    output = tf.reduce_sum(inputs * tf.expand_dims(vu, -1), 1)

    return output, alphas

