import cv2 as cv
import numpy as np
import sys
import math
import os
import os.path
import platform
from shutil import copy2, rmtree

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import detect_face

def filter_classes(data_path, output_path, threshold:int=20):
    """Filter person have images less than threshold"""

    # if platform.system() == "Windows":
    #     data_path = "D:\\Database\\CASIA-WebFace"
    #     output_path = "D:\\Database\\CASIA-WebFace_filtered_" + str(threshold)
    # elif platform.system() == "Linux":
    #     data_path = "/home/x000000/LK/Database/CASIA-WebFace"
    #     output_path = "/home/x000000/LK/Database/CASIA-WebFace_filtered"

    if os.path.exists(output_path):
        sys.stdout.write("Deleting...")
        sys.stdout.flush()
        rmtree(output_path)
        print()
    os.mkdir(output_path)

    cnt = 0
    for i, person in enumerate(os.listdir(data_path)):
        path = os.path.join(data_path, person)
        if len(os.listdir(path)) > threshold:
            out = os.path.join(output_path, str(cnt).zfill(7))
            os.mkdir(out)
            for img in os.listdir(path):
                file = os.path.join(path, img)
                copy2(file, out)
            cnt += 1
        sys.stdout.write("\r")
        sys.stdout.write("%s/%s" % (str(i+1), str(len(os.listdir(data_path)))))
        sys.stdout.flush()


def crop_face(data_path, output_path):
    """Crop face and output croped face image"""

    # if platform.system() == "Windows":
    #     data_path = "D:\\Database\\CASIA-WebFace"
    #     output_path = "D:\\Database\\CASIA-WebFace_croped"
    # elif platform.system() == "Linux":
    #     data_path = "/home/x000000/LK/Database/CASIA-WebFace_filtered"
    #     output_path = "/home/x000000/LK/Database/CASIA-WebFace_croped"

    if os.path.exists(output_path):
        sys.stdout.write("Deleting...")
        sys.stdout.flush()
        rmtree(output_path)
        print()
    os.mkdir(output_path)

    # load model
    gpu_memory_fraction=1.0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './data')

    minsize = 50  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # start crop
    file_ext = ".jpg"
    for i, person in enumerate(os.listdir(data_path)):
        cnt = 1
        path = os.path.join(data_path, person)
        # print(path)
        out = os.path.join(output_path, str(i).zfill(7))
        os.mkdir(out)
        for img in os.listdir(path):
            file = cv.imread(os.path.join(path, img))
            bounding_boxes, points = detect_face.detect_face(file, minsize, pnet, rnet, onet, threshold, factor)
            # num of face
            # num_faces = bounding_boxes.shape[0]
            # if num_faces == 0:
            #     continue
            try:
                bn = 0
                long_side = 0
                for l, box in enumerate(bounding_boxes):
                    box_width = face_pos[3] - face_pos[1]
                    box_height = face_pos[2] - face_pos[0]
                    box_longside = box_width if box_width > box_height else box_height
                    if box_longside > long_side:
                        long_side = box_longside
                        bn = l
                face_pos = bounding_boxes[bn]
                face_pos = face_pos.astype(int)
                # find the biggest bounding box
                # for box in bounding_boxes[1:]:
                #     box = box.astype(int)
                #     if ((box[3]-box[1]) * (box[2]-box[0]) >
                #         (face_pos[3]-face_pos[1]) * (face_pos[2]-face_pos[0])):
                #         face_pos = box
                crop_width = face_pos[3] - face_pos[1]
                crop_height = face_pos[2] - face_pos[0]
                long_side = crop_width if crop_width >= crop_height else crop_height
                face = file[face_pos[1]:face_pos[1]+long_side, face_pos[0]:face_pos[0]+long_side,]
                face = cv.resize(face, (250, 250))
            except:
                face = file[50:200, 50:200, ]
                face = cv.resize(face, (250, 250))
                cv.imwrite(os.path.join(out, str(cnt).zfill(3))+file_ext, face)
                cnt += 1
            else:
                cv.imwrite(os.path.join(out, str(cnt).zfill(3))+file_ext, face)
                cnt += 1
        sys.stdout.write("\r")
        sys.stdout.write("%s/%s" % (str(i+1), str(len(os.listdir(data_path)))))
        sys.stdout.flush()

def augenment_data(data_path, output_path):
    """Augenment Webface dataset"""

    # if platform.system() == "Windows":
    #     data_path = "D:\\Database\\CASIA-WebFace_croped"
    #     output_path = "D:\\Database\\CASIA-WebFace_augmented"
    # elif platform.system() == "Linux":
    #     data_path = "/home/x000000/LK/Database/CASIA-WebFace_filtered"
    #     output_path = "/home/x000000/LK/Database/CASIA-WebFace_croped"

    datagen = ImageDataGenerator(rotation_range=20,
                                 horizontal_flip=True)


def webface_preprocess(filter_flag:bool=False, filter_threshold:int=30,
                       crop_flag:bool=False, augenment_flag:bool=False):
    webface_filter = filter_flag and filter_classes()
    webface_crop = crop_flag and crop_face()
    webface_augment = augenment_flag and augenment_data()
    if webface_filter:
        webface_filter(filter_threshold)
    if webface_crop:
        webface_crop()
    if webface_augment:
        webface_augment()


if __name__ == "__main__":
    if platform.system() == "Windows":
        data_path = "D:\\Database\\CASIA-WebFace"
        output_path = "D:\\Database\\CASIA-WebFace"
    elif platform.system() == "Linux":
        data_path = "/home/x000000/LK/Database/CASIA-WebFace"
        output_path = "/home/x000000/LK/Database/CASIA-WebFace"

    # filter_th = 20
    # filter_th_str = str(filter_th) if filter_th else ""
    # filter_classes(data_path, output_path+"_filtered_"+filter_th_str, filter_th)
    # crop_face(data_path+"_filtered_"+filter_th_str, output_path+"_croped_"+filter_th_str)
    crop_face(data_path, output_path+"_croped")