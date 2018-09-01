import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

PI = 3.1415926535897932384626433

def a_softmax(feature, targets, num_outputs, m=2, lamb=1, name="a_softmax"):

    epsilon = 1e-8
    feature_shape = feature.get_shape()
    kernel = tf.get_variable("a_softmax/W", [feature_shape[1], num_outputs], dtype=tf.float32,
                             initializer=initializers.xavier_initializer())
    feature_norm = tf.nn.l2_normalize(feature, dim=1)
    kernel_norm = tf.nn.l2_normalize(kernel, dim=0)
    cos_theta = tf.matmul(feature_norm, kernel_norm) + epsilon

    logits = tf.matmul(feature, kernel_norm)

    if m == 1:
        adjust_logits = logits
        return adjust_logits
    elif m == 2:
        # cos_mtheta = 2*tf.pow(cos_theta, 2) - 1
        phi_theta = 2*tf.multiply(tf.sign(cos_theta), tf.square(cos_theta)) - 1
    # elif m == 3:
    #     cos_mtheta = 4*tf.pow(cos_theta, 3) - 3*cos_theta
    elif m == 4:
        # cos_mtheta = 8*tf.pow(cos_theta, 4) - 8*tf.pow(cos_theta) + 1
        cos_theta_square = tf.square(cos_theta)
        cos_theta_biquadrate = tf.pow(cos_th, 4)
        sign0 = tf.sign(cos_theta)
        sign3 = tf.multiply(tf.sign(2*cos_theta_square - 1), sign0)
        sign4 = 2*sign0 + sign3 - 3
        phi_theta = sign3*(8*cos_theta_biquadrate - 8*cos_theta_square + 1) + sign4

    adjust_theta = phi_theta # (lamb*cos_theta + phi_theta) / (1+lamb)
    logits_ = tf.multiply(tf.div(tf.matmul(feature, kernel_norm), cos_theta), adjust_theta)
    adjust_logits = tf.where(tf.equal(targets, 1), logits_, logits)

    return adjust_logits