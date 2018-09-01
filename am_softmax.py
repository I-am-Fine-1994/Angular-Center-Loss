import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

PI = 3.1415926535897932384626433

def am_softmax(feature, targets, num_outputs, s=32, m=0.35, name="am_softmax"):

    epsilon = 1e-8
    feature_shape = feature.get_shape()
    kernel = tf.get_variable("am_softmax/W", [feature_shape[1], num_outputs], dtype=tf.float32,
                             initializer=initializers.xavier_initializer())
    feature_norm = tf.nn.l2_normalize(feature, dim=1)
    kernel_norm = tf.nn.l2_normalize(kernel, dim=0)
    cos_theta = tf.matmul(feature_norm, kernel_norm)
    phi_theta = cos_theta - m

    logits = s * cos_theta
    logits_ = s * phi_theta

    adjust_logits = tf.where(tf.equal(targets, 1.0), logits_, logits)

    return adjust_logits