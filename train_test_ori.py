import sys
import platform
import shutil
import getpass
import os
import os.path

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

from webface_reader import WebFaceReader
from resnet34 import ResNet34


def model(feature, image_size, classes):
    inputs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    targets = tf.placeholder(tf.float32, [None, classes])
    is_training = tf.placeholder(tf.bool, [])

    x = ResNet34(inputs, is_training)
    x = slim.fully_connected(x, feature, scope="feature")
    logits = slim.fully_connected(x, classes, activation_fn=None, scope="logits")

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    cross_entropy = tf.losses.softmax_cross_entropy(targets, logits)

    step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_step = slim.learning.create_train_op(cross_entropy, optimizer, global_step=step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

    # Add summaries for BN variables
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("cross_entropy", cross_entropy)
    # for v in tf.all_variables():
    #     if v.name.startswith('conv1/Batch') or v.name.startswith('conv2/Batch') or \
    #             v.name.startswith('fc1/Batch') or v.name.startswith('logits/Batch'):
    #         print(v.name)
    #         tf.summary.histogram(v.name, v)
    merged_summary_op = tf.summary.merge_all()

    return {"inputs": inputs,
            "targets": targets,
            "is_training": is_training,
            "train_step": train_step,
            "global_step": step,
            "accuracy": accuracy,
            "cross_entropy": cross_entropy,
            "summary": merged_summary_op}

def train(model_id=0, feature=512, epochs=20, batch_size=32, val_split=0.1, image_size=250, data_path=""):
    print("Clearing existed checkpoint and logs")

    # logs_folder = "logs"+str(model_id).zfill(3)
    checkpoint_folder = str(model_id).zfill(3)
    # if os.path.exists(logs_folder):
    #     shutil.rmtree(logs_folder)
    if os.path.exists(checkpoint_folder):
        shutil.rmtree(checkpoint_folder)
    os.mkdir(checkpoint_folder)

    # if all_folder_flag:
    classes = len(os.listdir(data_path))

    data_reader = WebFaceReader(data_path=data_path,
                                val_split=val_split,
                                batch_size=batch_size,
                                image_size=image_size,
                                classes=classes)
    net = model(feature=feature, image_size=image_size, classes=classes)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(os.path.join(checkpoint_folder, 'train'), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(checkpoint_folder, 'valid'), sess.graph)

    print("model_id=%03d\nnetwork=%s\nbatch_size=%d\nval_split=%0.2f\nimage_size=%d\nclasses=%d\ndata_path=%s"
          % (model_id, "resnet34", batch_size, val_split, image_size, classes, data_path))

    saver.save(sess, os.path.join(checkpoint_folder, "model"), write_meta_graph=False)

    for epoch in range(epochs):
        print("Epoch %d/%d" % (epoch+1, epochs))
        # Training
        training_steps = len(data_reader.train_seq)
        for i in range(training_steps):
            batch_x, batch_y = data_reader.train_seq[i]
            train_dict ={net["inputs"]:batch_x,
                         net["targets"]:batch_y,
                         net["is_training"]:True}
            entropy, acc, global_step, train_step, summary = sess.run(
                                                            [net["cross_entropy"],
                                                             net["accuracy"],
                                                             net["global_step"],
                                                             net["train_step"],
                                                             net["summary"]],
                                                            feed_dict=train_dict)
            if global_step % 50 == 0:
                # entropy, acc, global_step, summary = sess.run(
                #                                             [net["cross_entropy"],
                #                                              net["accuracy"],
                #                                              net["global_step"],
                #                                              net["summary"]],
                #                                             feed_dict=train_dict)
                train_writer.add_summary(summary, global_step=global_step)

            sys.stdout.write("\r")
            sys.stdout.write(" "*100)
            sys.stdout.write("\r")
            sys.stdout.write("    Train step %d/%d: entropy %f: accuracy %0.4f" % (int(i+1), training_steps, entropy, acc))
            sys.stdout.flush()

        # Validation
        validation_steps = len(data_reader.val_seq)
        val_acc = 0
        for j in range(validation_steps):
            batch_xv, batch_yv = data_reader.val_seq[i]
            train_dict ={net["inputs"]:batch_xv,
                         net["targets"]:batch_yv,
                         net["is_training"]:False}
            val_entropy, val_acc_, val_summary = sess.run([
                                                    net['cross_entropy'],
                                                    net['accuracy'],
                                                    net['summary']],
                                                    feed_dict=train_dict)
            val_acc += val_acc_
        val_acc /= validation_steps
        valid_writer.add_summary(summary, global_step=global_step)
        print('\n***** Valid step {}: entropy {}: accuracy {} *****'.format(global_step, val_entropy, val_acc))
        saver.save(sess, os.path.join(checkpoint_folder, "epoch: %d" % epoch), write_meta_graph=False)

    sess.close()
    return

# def test(model_id):
#     logs_folder = "logs"+str(model_id).zfill(3)
#     checkpoint_folder = str(model_id).zfill(3)

#     # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#     # Test trained model
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.import_meta_graph()

if __name__ == "__main__":
    username = getpass.getuser()
    if platform.system() == "Windows":
        webface_data_path = "D:\\Database\\CASIA-WebFace"
    elif platform.system() == "Linux":
        webface_data_path = "/home/"+username+"/LK/Database/CASIA-WebFace"
    train(model_id=0, data_path=webface_data_path+"_croped_20")
    # test(model_id=0)