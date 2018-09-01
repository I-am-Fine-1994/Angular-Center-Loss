import tensorflow as tf
import tensorflow.contrib.slim as slim

# weights_regularizer = slim.l2_regularizer(0.0001)

def building_block(inputs, filters, stage, block, downsample=False, is_training=True):

    block_scope_name = "building_block" + str(stage) + block

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer = slim.l2_regularizer(0.0001),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={"is_training": is_training,
                                           "decay": 0.9,
                                           "scale": True},
                        scope = block_scope_name):
        if downsample:
            x = slim.conv2d(inputs, filters, [3, 3], 2, scope=block_scope_name+"_conv1_ds")
            shortcut = slim.conv2d(inputs, filters, [3, 3], 2, normalizer_fn=None, scope=block_scope_name+"_conv1_sc")
        else:
            x = slim.conv2d(inputs, filters, [3, 3], 1, scope=block_scope_name+"_conv1")
            shortcut = inputs
        x = slim.conv2d(x, filters, [3, 3], 1, activation_fn=None, scope=block_scope_name+"_conv2")
        x = tf.add(x, shortcut)
        x = tf.nn.relu(x)

        return x

def ResNet34_L2(inputs, is_training):

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer = slim.l2_regularizer(0.0001),
                        scope="resnet34"):
        x = slim.conv2d(inputs, 64, [7, 7], 2, scope="conv1")
        x = slim.max_pool2d(x, [3, 3])

        x = building_block(x, 64, 2, "a", is_training=is_training)
        x = building_block(x, 64, 2, "b", is_training=is_training)
        x = building_block(x, 64, 2, "c", is_training=is_training)

        x = building_block(x, 128, 3, "a", downsample=True, is_training=is_training)
        x = building_block(x, 128, 3, "b", is_training=is_training)
        x = building_block(x, 128, 3, "c", is_training=is_training)
        x = building_block(x, 128, 3, "d", is_training=is_training)


        x = building_block(x, 256, 4, "a", downsample=True, is_training=is_training)
        x = building_block(x, 256, 4, "b", is_training=is_training)
        x = building_block(x, 256, 4, "c", is_training=is_training)
        x = building_block(x, 256, 4, "d", is_training=is_training)
        x = building_block(x, 256, 4, "e", is_training=is_training)
        x = building_block(x, 256, 4, "f", is_training=is_training)

        x = building_block(x, 512, 5, "a", downsample=True, is_training=is_training)
        x = building_block(x, 512, 5, "b", is_training=is_training)
        x = building_block(x, 512, 5, "c", is_training=is_training)

        x = slim.avg_pool2d(x, [8, 8])
        x = slim.flatten(x)

        return x