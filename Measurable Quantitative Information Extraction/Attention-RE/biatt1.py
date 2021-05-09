import tensorflow as tf


def attention1(inputs):
    hidden_size = inputs.shape[2].value
    u_omega = tf.get_variable("u_omega1", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

    with tf.name_scope('v'):
        v =inputs

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')
    alphas = tf.nn.softmax(vu, name='alphas')

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output, alphas

