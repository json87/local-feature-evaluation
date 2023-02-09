# Revised 2020-7-27: San Jiang <jiangsan@cug.edu.cn>
#  - Make this script adapt to the revision of the master COLMAP.
#  - Add some SOTA learned local feature descriptors.
#
# matchnet - https://github.com/hanxf/matchnet

from __future__ import print_function

import os
import sys
import math
import time
import argparse
import numpy as np
import h5py

from caffe.proto import caffe_pb2
from matchnet import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the dataset, e.g., path/to/Fountain")
    parser.add_argument("--model_path", required=True,
                        help="Path to the matchnet model definition file")
    parser.add_argument("--params_path", required=True,
                        help="Path to the matchnet pretained parameter file")
    parser.add_argument("--batch_size", type=int, default=512)
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


def main():
    args = parse_args()

    # Initialize networks.
    caffe.set_mode_gpu()
    feature_net = FeatureNet(args.model_path, args.params_path)

    if not os.path.exists(os.path.join(args.dataset_path, "descriptors")):
        os.makedirs(os.path.join(args.dataset_path, "descriptors"))

    # image_names = os.listdir(os.path.join(args.dataset_path, "images"))
    image_names = []
    list_dir_recursive(os.path.join(args.dataset_path, "images"), image_names)

    for i, image_name in enumerate(image_names):
        seperator_pos = len(os.path.join(args.dataset_path,
                                         "images").replace('\\', '/')) + 1
        image_name_ex = image_name[seperator_pos:].replace('/', '-')
        patches_path = os.path.join(args.dataset_path, "descriptors",
                                    image_name_ex + ".bin.patches.mat")

        if not os.path.exists(patches_path):
            continue

        print("Computing features for {} [{}/{}]".format(
              image_name, i + 1, len(image_names)), end="")

        start_time = time.time()

        descriptors_path = os.path.join(args.dataset_path, "descriptors",
                                        image_name_ex + ".bin")
        if os.path.exists(descriptors_path):
            print(" -> skipping, already exist")
            continue

        with h5py.File(patches_path, 'r') as patches_file:
            patches31 = np.array(patches_file["patches"]).T

        if patches31.ndim != 3:
            print(" -> skipping, invalid input")
            write_matrix(descriptors_path, np.zeros((0, 128), dtype=np.float32))
            continue

        patches = np.empty((patches31.shape[0], 64, 64), dtype=np.float32)
        patches[:, :63, :63] = patches31
        patches[:, 63, :63] = patches31[:, 62, :]
        patches[:, :63, 63] = patches31[:, :, 62]
        patches[:, 63, 63] = patches31[:, 62, 62]
        patches = np.expand_dims(patches, axis=1)

        descriptors = []
        for i in range(0, patches.shape[0], args.batch_size):
            patches_batch = \
                patches[i:min(i + args.batch_size, patches.shape[0])]
            descriptors_batch = feature_net.ComputeFeature(patches_batch)
            descriptors.append(descriptors_batch.reshape(-1, 4096))

        if len(descriptors) == 0:
            descriptors = np.zeros((0, 128), dtype=np.float32)
        else:
            descriptors = np.concatenate(descriptors)

        write_matrix(descriptors_path, descriptors)

        print(" in {:.3f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
