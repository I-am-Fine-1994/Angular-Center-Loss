import sys
import platform
import shutil
import getpass
import os
import os.path
import math
import time
import logging
import getpass

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import sklearn.metrics.pairwise as pw

from webface_reader import WebFaceReader
from resnet34 import ResNet34
from resnet34_l2_regularizer import ResNet34_L2

from l2_softmax import l2_softmax
from l_softmax import l_softmax
from am_softmax import am_softmax
from a_softmax import a_softmax
from arcface_softmax import arcface_softmax
from angular_center_loss import angular_center_loss

from center_loss import get_center_loss, center_loss
from weight_center_loss import weight_center_loss

def initLogging(filename, logger_name):
    logging.basicConfig(level=logging.INFO,
                        filemode="w",
                        format="%(message)s")

# def network(inputs, is_training):
#     x = ResNet34(inputs, is_training)
#     x = slim.fully_connected(x, feature, scope="feature")
#     logits = slim.fully_connected(x, classes, activation_fn=None, scope="logits")
#     return logits

def model(feature_dim, image_size, classes):
    inputs = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name="inputs")
    targets = tf.placeholder(tf.float32, [None, classes], name="targets")
    is_training = tf.placeholder(tf.bool, [], name="is_training")
    learning_rate = tf.Variable(tf.constant(0.01), trainable=False, dtype=tf.float32, name="learning_rate")
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.1)

    # logits = network(inputs, is_training)
    x = ResNet34(inputs, is_training)
    feature = slim.fully_connected(x, feature_dim, scope="feature")

    if model_id in [33, 34, 40, 47]:
        # model_id = 33, 34 fully_connected
        logits = slim.fully_connected(feature, classes, activation_fn=None, scope="logits")
    elif model_id in [35, 45]:
        # model_id = 35, l2-softmax logits
        logits = l2_softmax(feature, classes)
    elif model_id in [36]:
        # model_id = 36, l-softmax logits
        logits = l_softmax(feature, targets, classes)
    elif model_id in [37]:
        # model_id = 37, a-softmax logits
        logits = a_softmax(feature, targets, classes)
    elif model_id in [38, 46]:
        # model_id = 38, am-softmax logits
        logits = am_softmax(feature, targets, classes, s=32, m=0.1)
    elif model_id in [41]:
        # model_id = 041, weight_center_loss
        logits, loss = weight_center_loss(feature, targets, classes)
    elif model_id in [43, 44]:
        # model_id = 043, arcface_softmax
        logits = arcface_softmax(feature, targets, classes, s=32, m=0.1)

    logits = tf.clip_by_value(tf.nn.softmax(logits), 1e-10, 1.0)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(targets*tf.log(logits), 1))
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    # cross_entropy = tf.losses.softmax_cross_entropy(targets, logits)
    if model_id in [40, 44, 45, 46]:
        loss, centers = center_loss(feature, targets, 0.5, classes)
    if model_id in [47]:
        loss, centers = angular_center_loss(feature, targets, classes)
    if model_id in [40, 41, 44, 45, 46, 47]:
        entropy = tf.add(cross_entropy, 0.01*loss)
    else:
        entropy = cross_entropy

    step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    train_step = slim.learning.create_train_op(entropy, optimizer, global_step=step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

    tf.summary.scalar("accuracy", accuracy)
    if model_id in [40, 41, 44, 45, 46]:
        tf.summary.scalar("cross entropy", cross_entropy)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("total loss", entropy)
    else:
        tf.summary.scalar("cross entropy", cross_entropy)
    merged_summary_op = tf.summary.merge_all()

    return {"inputs": inputs,
            "targets": targets,
            "is_training": is_training,
            "train_step": train_step,
            "global_step": step,
            "accuracy": accuracy,
            "cross_entropy": cross_entropy,
            "summary": merged_summary_op,
            "feature": feature,
            "lr_decay": learning_rate_decay_op,
            "lr": learning_rate}

def train(model_id=0, id_suffix=0, feature_dim=512, epochs=20, batch_size=32, val_split=0.1, image_size=250, classes=7211, decay_list = [10, 15], data_path=""):

    print("Clearing existed checkpoint and logs")
    checkpoint_folder = str(model_id).zfill(3)+"_"+str(id_suffix).zfill(2)
    while os.path.exists(checkpoint_folder):
        del_confirm = input("You are gonna delete a existed model record %s, are you sure? (y/n)" % checkpoint_folder)
        # del_confirm = "y"
        if del_confirm == "y":
            shutil.rmtree(checkpoint_folder)
            print("model record %s was removed." % checkpoint_folder)
            break
        elif del_confirm == "n":
            model_id, id_suffix = input("Now you can set the model_id: ")
            model_id = int(model_id)
            id_suffix = int(id_suffix)
            checkpoint_folder = str(model_id).zfill(3)+"_"+str(id_suffix).zfill(2)
        else:
            del_confirm = input("You need to enter 'y' or 'n': ")
    os.mkdir(checkpoint_folder)

    recording_file = os.path.join(checkpoint_folder, "print.txt")
    recording = open(recording_file, "w")

    # classes = 100 # len(os.listdir(data_path))

    data_reader = WebFaceReader(data_path=data_path,
                                val_split=val_split,
                                batch_size=batch_size,
                                image_size=image_size,
                                classes=classes)
    net = model(feature_dim=feature_dim, image_size=image_size, classes=classes)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(os.path.join(checkpoint_folder, 'train'), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(checkpoint_folder, 'valid'), sess.graph)

    print("experiment_id=%s\nnetwork=%s\nbatch_size=%d\nval_split=%0.2f\nimage_size=%d\nclasses=%d\ndata_path=%s"
          % (checkpoint_folder, "resnet34", batch_size, val_split, image_size, classes, data_path))
    print("experiment_id=%s\nnetwork=%s\nbatch_size=%d\nval_split=%0.2f\nimage_size=%d\nclasses=%d\ndata_path=%s"
          % (checkpoint_folder, "resnet34", batch_size, val_split, image_size, classes, data_path), file=recording)

    saver.save(sess, os.path.join(checkpoint_folder, "model"))

    start = time.time()
    best_val_acc = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch %d/%d" % (epoch+1, epochs))
        print("Epoch %d/%d" % (epoch+1, epochs), file=recording)
        if epoch+1 in decay_list:
            print("learning_rate reduced.")
            print("learning_rate reduced.", file=recording)
            sess.run(net["lr_decay"])
        lr = sess.run(net["lr"])
        print("learning rate is %f" % lr)
        print("learning rate is %f" % lr, file=recording)
        # Training
        train_acc = 0
        train_entropy = 0
        training_steps = len(data_reader.train_seq)
        for i in range(training_steps):
            epoch_eta = time.time()
            epoch_eta -= epoch_start
            epoch_eta = epoch_eta/(i+1) * (training_steps-i)

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
            train_acc += acc
            train_entropy += entropy
            if global_step % 50 == 0:
                train_writer.add_summary(summary, global_step=global_step)

            sys.stdout.write("\r")
            sys.stdout.write(" "*100)
            sys.stdout.write("\r")
            sys.stdout.write("ETA: %02d:%02d:%02d Train step %d/%d: entropy %f: accuracy: %0.4f"
                             % (epoch_eta//3600, (epoch_eta%3600)//60, epoch_eta%60, int(i+1), training_steps, train_entropy/(i+1), (train_acc/(i+1))))
            sys.stdout.flush()
        # print(" centers_l2: %f" % centers)
        sys.stdout.write("\r")

        # Validation
        validation_steps = len(data_reader.val_seq)
        val_acc = 0
        val_entropy = 0
        for j in range(validation_steps):
            batch_xv, batch_yv = data_reader.val_seq[j]
            train_dict ={net["inputs"]:batch_xv,
                         net["targets"]:batch_yv,
                         net["is_training"]:False}
            val_entropy_, val_acc_, val_summary = sess.run([
                                                    net['cross_entropy'],
                                                    net['accuracy'],
                                                    net['summary']],
                                                    feed_dict=train_dict)
            val_acc += val_acc_
            val_entropy += val_entropy_
        # valid_writer.add_summary(val_summary, global_step=(i+1)*training_steps)

        train_acc /= training_steps
        train_entropy /= training_steps
        val_acc /= validation_steps
        val_entropy /= validation_steps
        saver.save(sess, os.path.join(checkpoint_folder, "epoch_%02d" % (epoch+1)), write_meta_graph=False)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            saver.save(sess, os.path.join(checkpoint_folder, "best_val_acc"), write_meta_graph=False)

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        print("Time cost: %02d:%02d:%02d train_entropy: %f train_accuracy: %0.4f val_entropy: %f val_accuracy: %0.4f"
              % (epoch_duration//3600, (epoch_duration%3600)//60, epoch_duration%60, train_entropy, train_acc, val_entropy, val_acc))
        print("Time cost: %02d:%02d:%02d train_entropy: %f train_accuracy: %0.4f val_entropy: %f val_accuracy: %0.4f"
              % (epoch_duration//3600, (epoch_duration%3600)//60, epoch_duration%60, train_entropy, train_acc, val_entropy, val_acc), file=recording)
    print()

    end = time.time()
    duration = end - start
    print("Training Time Cost: %02d:%02d:%02d" % (duration//3600, (duration%3600)//60, duration%60))
    print("Training Time Cost: %02d:%02d:%02d" % (duration//3600, (duration%3600)//60, duration%60), file=recording)

    saver.save(sess, os.path.join(checkpoint_folder, "model"))

    sess.close()
    recording.close()
    return

def test(model_id, id_suffix=0):

    checkpoint_folder = str(model_id).zfill(3) + "_" + str(id_suffix).zfill(2)
    print(checkpoint_folder)

    recording_file = os.path.join(checkpoint_folder, "result.txt")
    recording = open(recording_file, "w")

    # initLogging(os.path.join(checkpoint_folder, "result.txt"), "test_logger")
    # test_logger = logging.getLogger("test_logger")
    # test_logger.addHandler(logging.StreamHandler())

    # Test trained model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # net = model()
    saver = tf.train.import_meta_graph(os.path.join(checkpoint_folder, "model.meta"))
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_folder))
    # saver.restore(sess, os.path.join(checkpoint_folder, "best_val_acc"))

    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name("inputs:0")
    # targets = graph.get_tensor_by_name("targets:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    feature = graph.get_tensor_by_name("feature/Relu:0")

    username = getpass.getuser()
    if platform.system() == "Windows":
        img_path = "D:\\Database\\lfw_6000_pair"
    elif platform.system() == "Linux":
        img_path = "/home/" + username + "/LK/Database/lfw_6000_pair"

    label = []
    data = []
    img_ext = ".jpg"
    for i in range(6000):
        data.append([os.path.join(img_path, str(i).zfill(4)+"_"+"0"+img_ext),
                     os.path.join(img_path, str(i).zfill(4)+"_"+"1"+img_ext)])
        label.append(0 if (i//300)%2 else 1)
    # print(label[0].shape)

    print("Test on model %s" % checkpoint_folder)
    print("Test on model %s" % checkpoint_folder, file=recording)
    start = time.time()
    predict = []
    num_of_pairs = len(data)
    for i, pair in enumerate(data):
        face1 = cv.imread(pair[0])
        face2 = cv.imread(pair[1])
        face1 = cv.resize(face1, (250, 250))
        face2 = cv.resize(face2, (250, 250))
        face1 = np.reshape(face1, (1, 250, 250, 3))
        face2 = np.reshape(face2, (1, 250, 250, 3))
        feed_dict = {inputs: face1, is_training: False}
        feature1 = sess.run(feature, feed_dict=feed_dict)
        feed_dict = {inputs: face2, is_training: False}
        feature2 = sess.run(feature, feed_dict=feed_dict)
        pred = pw.cosine_similarity(feature1,feature2)
        predict.append(np.squeeze(pred[0]))

        sys.stdout.write("\r")
        sys.stdout.write("%s/%s" % (str(i+1), str(num_of_pairs)))
        sys.stdout.flush()
    print()

    start_th = 0.1
    end_th = 0.9
    stair_step = 0.01
    stairs = math.ceil((end_th-start_th)/stair_step-0.5) + 1
    best_acc = 0
    best_threhold = 0
    for i in range(stairs):
        threshold = start_th + stair_step*i
        print()
        print("threshold=%0.2f" % threshold)
        print("threshold=%0.2f" % threshold, file=recording)
        hits = 0
        tp = fn = fp = tn = 0
        far_points = frr_points = acc_points = []
        predict_th = [1 if p>threshold else 0 for p in predict]
        for tlbl, plbl in zip(label, predict_th):
            if tlbl == 1 and plbl == 1:
                tp += 1
            elif tlbl == 1 and plbl == 0:
                fn += 1
            elif tlbl == 0 and plbl == 1:
                fp += 1
            elif tlbl == 0 and plbl == 0:
                tn += 1

        hits = tp + tn
        far = fp/(fp + tn)
        frr = fn/(fn + tp)
        acc = hits/num_of_pairs
        if acc > best_acc:
            best_acc = acc
            best_threhold = threshold
        far_points.append((threshold, far))
        frr_points.append((threshold, frr))
        acc_points.append((threshold, acc))

        print("confusion mattrix:\n%d\t%d\n%d\t%d" % (tp, fn, fp, tn))
        print("ACC on lfwcrop_color pairs is %d/%d = %0.4f" % (hits, num_of_pairs, acc))
        print("FAR on lfwcrop_color pairs is %d/%d = %0.4f" % (fp, (fp+tn), far))
        print("FRR on lfwcrop_color pairs is %d/%d = %0.4f" % (fn, (fn+tp), frr))
        print("confusion mattrix:\n%d\t%d\n%d\t%d" % (tp, fn, fp, tn), file=recording)
        print("ACC on lfwcrop_color pairs is %d/%d = %0.4f" % (hits, num_of_pairs, acc), file=recording)
        print("FAR on lfwcrop_color pairs is %d/%d = %0.4f" % (fp, (fp+tn), far), file=recording)
        print("FRR on lfwcrop_color pairs is %d/%d = %0.4f" % (fn, (fn+tp), frr), file=recording)
    print()

    end = time.time()
    duration = end - start
    print("Model %s test Time Cost: %02d:%02d:%02d" % (checkpoint_folder, duration//3600, (duration%3600)//60, duration%60))
    print("Best accuracy is %04f on threshold %02f" % (best_acc, best_threhold))
    print("Model %s test Time Cost: %02d:%02d:%02d" % (checkpoint_folder, duration//3600, (duration%3600)//60, duration%60), file=recording)
    print("Best accuracy is %04f on threshold %02f" % (best_acc, best_threhold), file=recording)
    recording.close()
    sess.close()


if __name__ == "__main__":
    username = getpass.getuser()
    if platform.system() == "Windows":
        webface_data_path = "D:\\Database\\CASIA-WebFace"
    elif platform.system() == "Linux":
        webface_data_path = "/home/"+username+"/LK/Database/CASIA-WebFace"

    model_id, id_suffix, classes, epochs = 47, 5, 7211, 20
    decay_list = [10, 15]
    train(model_id=model_id, id_suffix=id_suffix, classes=classes, data_path=webface_data_path+"_croped_20", epochs=epochs, decay_list=decay_list)
    test(model_id=model_id, id_suffix=id_suffix)