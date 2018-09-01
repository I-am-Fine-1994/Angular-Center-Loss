import tensorflow as tf
import math
from tensorflow.contrib.layers.python.layers import initializers

PI = 3.1415926535897932384626433

def arcface_softmax(feature, targets, num_outputs, s=32, m=0.5, name="arcface_softmax"):

    feature_shape = feature.get_shape()
    kernel = tf.get_variable("arcface_softmax/W", [feature_shape[1], num_outputs], dtype=tf.float32,
                             initializer=initializers.xavier_initializer())
    feature_norm = tf.nn.l2_normalize(feature, dim=1)
    kernel_norm = tf.nn.l2_normalize(kernel, dim=0)
    cos_theta = tf.matmul(feature_norm, kernel_norm)

    theta = tf.acos(cos_theta)
    phi_theta = tf.cos(theta + m)

    # cos_m = math.cos(m)
    # sin_m = math.sin(m)
    # cos_theta2 = tf.square(cos_theta)
    # sin_theta2 = tf.subtract(1.0, cos_theta2)
    # sin_theta = tf.sqrt(sin_theta2)
    # phi_theta = tf.subtract(tf.multiply(cos_theta, cos_m), tf.multiply(sin_theta, sin_m))

    logits = s * cos_theta
    logits_ = s * phi_theta

    adjust_logits = tf.where(tf.equal(targets, 1.0), logits_, logits)

    return adjust_logits