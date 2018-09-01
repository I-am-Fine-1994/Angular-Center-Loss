import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

def l2_softmax(inputs, num_outputs, alpha=20, name="l2_softmax"):
    inputs_shape = inputs.get_shape()

    # alpha = tf.constant(alpha, dtype=tf.float32, name="alpha")
    kernel = tf.get_variable("l2_softmax/W", [inputs_shape[1], num_outputs], dtype=tf.float32,
                             initializer=initializers.xavier_initializer())

    inputs = tf.nn.l2_normalize(inputs, dim=1)
    inputs = alpha * inputs
    logits = tf.matmul(inputs, kernel)
    return logits