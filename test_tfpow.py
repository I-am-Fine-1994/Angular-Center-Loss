import tensorflow as tf

def test_tfpow():
    x = tf.constant([2, 2])
    y = tf.constant([[8, 16], [2, 3]])

    pow_op = tf.pow(x, y)  # [[256, 65536], [9, 27]]

    with tf.Session() as sess:
        print(sess.run(pow_op))
    return

if __name__ == "__main__":
    test_tfpow()