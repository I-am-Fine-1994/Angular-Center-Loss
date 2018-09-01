import tensorflow as tf

def test_l2_normalize():
    mat = tf.constant([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]])
    mat_norm = tf.nn.l2_normalize(mat, dim=(0, 1))

    with tf.Session() as sess:
        print(sess.run(mat_norm))
    return

if __name__ == "__main__":
    test_l2_normalize()