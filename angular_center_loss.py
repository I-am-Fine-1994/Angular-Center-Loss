import tensorflow as tf

def angular_center_loss(features, label, classes):

    label = tf.argmax(label, axis=1)
    label = tf.cast(label, dtype=tf.int64)
    feature_dim = features.get_shape()[1]
    centers = tf.get_variable("angular_centers", [classes, feature_dim], dtype=tf.float32,
        initializer = tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])

    centers_batch = tf.gather(centers, label)
    features_norm = tf.nn.l2_normalize(features, dim=1)
    centers_batch_norm = tf.nn.l2_normalize(centers_batch, dim=1)

    unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    update = tf.div(features_norm, tf.cast(appear_times, tf.float32))
    # updated_centers_batch = tf.add(centers_batch_norm, update)
    # updated_centers_batch_norm = tf.nn.l2_normalize(updated_centers_batch)

    centers = tf.scatter_add(centers, label, update)
    centers = tf.nn.l2_normalize(centers, dim=1)
    loss = -tf.reduce_sum(tf.multiply(features_norm, centers_batch_norm))

    return loss, centers




    # nrof_features = features.get_shape()[1]
    # #训练过程中，需要保存当前所有类中心的全连接预测特征centers， 每个batch的计算都要先读取已经保存的centers
    # centers = tf.get_variable('angular_centers', [nrof_classes, nrof_features], dtype=tf.float32,
    #     initializer=tf.constant_initializer(0), trainable=False)
    # label = tf.reshape(label, [-1])
    # centers_batch = tf.gather(centers, label)#获取当前batch对应的类中心特征
    # # diff = (1 - alfa) * (centers_batch - features)#计算当前的类中心与特征的差异，用于Cj的的梯度更新，这里facenet的作者做了一个 1-alfa操作，比较奇怪，和原论文不同

    # # 当前mini-batch的特征值与它们对应的中心值之间的差
    # diff = centers_batch - features

    # # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    # unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
    # appear_times = tf.gather(unique_count, unique_idx)
    # appear_times = tf.reshape(appear_times, [-1, 1])

    # diff = tf.div(diff, tf.cast((1 + appear_times), tf.float32))
    # diff = alfa * diff

    # centers = tf.scatter_sub(centers, label, diff)#更新梯度Cj，对于上图中步骤6，tensorflow会将该变量centers保留下来，用于计算下一个batch的centerloss
    # loss = tf.reduce_mean(tf.square(features - centers_batch))#计算当前的centerloss 对应于Lc
    # return loss, centers