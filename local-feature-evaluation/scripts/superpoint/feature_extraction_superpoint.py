# Revised 2020-8-3: San Jiang <jiangsan@cug.edu.cn>
#  - Make this script adapt to the revision of the master COLMAP.
#  - Add some SOTA learned local feature descriptors.
#
# superpoint - https://github.com/rpautrat/SuperPoint

import os
import sys
import math
import time
import argparse
import cv2
import h5py

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the dataset, e.g., path/to/Fountain")
    parser.add_argument("--model_path", required=True,
                        help="Path to the pretrained geodesc model")
    parser.add_argument('--H', type=int, default=1000,
                        help='The height in pixels to resize the images to. \
                                    (default: 1000)')
    parser.add_argument('--W', type=int, default=1496,
                        help='The width in pixels to resize the images to. \
                                    (default: 1496)')
    parser.add_argument('--k_best', type=int, default=8000,
                        help='Maximum number of keypoints to keep \
                            (default: 5000)')
    args = parser.parse_args()
    return args


def list_dir_recursive(path, files):
    for file in os.listdir(path):
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            list_dir_recursive(cur_path, files)
        else:
            files.append(cur_path.replace('\\', '/'))


def write_matrix(path, matrix):
    with open(path, "wb") as fid:
        shape = np.array(matrix.shape, dtype=np.int32)
        shape.tofile(fid)
        matrix.tofile(fid)


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[0:2]
    origin_longside = 0 if width > height else 1
    target_longside = 0 if img_size[0] > img_size[1] else 1
    if origin_longside != target_longside:
        img_size = (img_size[1], img_size[0])

    img_scale = (width/img_size[0], height/img_size[1])
    img_size_origin = (width, height)

    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_scale, img_size_origin


def extract_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                      # img_origin,
                                      img_scale, image_size_origin,
                                      keep_k_points=8000):

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]].astype(np.float32)

    keypoints = np.array([[min(p[1]*img_scale[0], image_size_origin[0]),
                  min(p[0]*img_scale[1], image_size_origin[1]),
                  1.0, 1.0] for p in keypoints]).astype(np.float32)

    #################################################################################
    # Convert from just pts to cv2.KeyPoints
    # kpts_cv2 = [cv2.KeyPoint(p[0], p[1], 1) for p in keypoints]
    # cv2.drawKeypoints(img_origin, kpts_cv2, img_origin, (0, 255, 255),
    # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    # cv2.imshow('img', img_origin)
    # cv2.waitKey(0)
    # cv2.imwrite('E:/1.jpg', img_origin)
    #################################################################################

    return keypoints, desc


def main():
    args = parse_args()

    img_size = (args.W, args.H)
    keep_k_best = args.k_best

    if not os.path.exists(os.path.join(args.dataset_path, "keypoints")):
        os.makedirs(os.path.join(args.dataset_path, "keypoints"))
    if not os.path.exists(os.path.join(args.dataset_path, "descriptors")):
        os.makedirs(os.path.join(args.dataset_path, "descriptors"))

    tf.debugging.set_log_device_placement(True)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   args.model_path)

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        # image_names = os.listdir(os.path.join(args.dataset_path, "images"))
        image_names = []
        list_dir_recursive(os.path.join(args.dataset_path, "images"), image_names)

        for i, image_name in enumerate(image_names):
            seperator_pos = len(os.path.join(args.dataset_path,
                                             "images").replace('\\', '/')) + 1
            image_name_ex = image_name[seperator_pos:].replace('/', '-')

            image_path = os.path.join(args.dataset_path, image_name)
            if not os.path.exists(image_path):
                continue

            print("Computing features and descriptors for {} [{}/{}]".format(
                  image_name, i + 1, len(image_names)), end="")

            start_time = time.time()

            keypoint_path = os.path.join(args.dataset_path, "keypoints",
                                         image_name_ex + ".bin")
            descriptors_path = os.path.join(args.dataset_path, "descriptors",
                                            image_name_ex + ".bin")
            if os.path.exists(keypoint_path) and \
                    os.path.exists(descriptors_path):
                print(" -> skipping, already exist")
                continue

            img, img_scale, img_size_origin = preprocess_image(image_path, img_size)
            out = sess.run([output_prob_nms_tensor, output_desc_tensors],
                           feed_dict={input_img_tensor: np.expand_dims(img, 0)})
            keypoint_map = np.squeeze(out[0])
            descriptor_map = np.squeeze(out[1])
            keypoints, descriptors = extract_keypoints_and_descriptors(
                keypoint_map, descriptor_map, img_scale, img_size_origin, keep_k_best)

            write_matrix(keypoint_path, keypoints)
            write_matrix(descriptors_path, descriptors)

            print(" in {:.3f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
