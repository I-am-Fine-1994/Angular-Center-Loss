import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

def weight_center_loss(feature, targets, num_outputs, s=32):

    targets_index = tf.argmax(targets, axis=1)
    feature_shape = feature.get_shape()
    kernel = tf.get_variable("weight_center_loss/W", [feature_shape[1], num_outputs], dtype=tf.float32,
                             initializer=initializers.xavier_initializer())

    feature_norm = tf.nn.l2_normalize(feature, dim=1)
    kernel_norm = tf.nn.l2_normalize(kernel, dim=0)

    transposed_kernel = s * tf.transpose(kernel_norm)
    weights_batch = tf.gather(transposed_kernel, targets_index)
    weight_center_loss = tf.nn.l2_loss(feature - weights_batch)

    logits = s * tf.matmul(feature_norm, kernel_norm)
    # logits = s * logits

    return logits, weight_center_loss