# Revised 2020-7-17: San Jiang <jiangsan@cug.edu.cn>
#  - Make this script adapt to the revision of the master COLMAP.
#  - Add some SOTA learned local feature descriptors.
#
# D2-NET : https://github.com/mihaidusmanu/d2-net

import os
import sys
import math
import time
import argparse
import h5py
import imageio
import numpy as np

import torch

from tqdm import tqdm
from PIL import Image

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the dataset, e.g., path/to/Fountain")
    parser.add_argument("--model_path", required=True,
                        help="Path to the pretained hardnet model")
    parser.add_argument('--preprocessing', type=str, default='caffe',
                        help='image preprocessing (caffe or torch)')
    parser.add_argument('--max_edge', type=int, default=8000,
                        help='maximum image size at network input')
    parser.add_argument('--max_sum_edges', type=int, default=16000,
                        help='maximum sum of image sizes at network input')
    parser.add_argument('--multiscale', dest='multiscale', action='store_true',
                        help='extract multiscale features')
    parser.set_defaults(multiscale=False)
    parser.add_argument('--no-relu', dest='use_relu', action='store_false',
                        help='remove ReLU after the dense feature extraction module')
    parser.set_defaults(use_relu=True)

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

    # CUDA
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Creating CNN model
    model = D2Net(
        model_file=args.model_path,
        use_relu=args.use_relu,
        use_cuda=use_cuda
    )

    if not os.path.exists(os.path.join(args.dataset_path, "keypoints")):
        os.makedirs(os.path.join(args.dataset_path, "keypoints"))
    if not os.path.exists(os.path.join(args.dataset_path, "descriptors")):
        os.makedirs(os.path.join(args.dataset_path, "descriptors"))

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

        keypoint_path = os.path.join(args.dataset_path, "keypoints",
                                     image_name_ex + ".bin")
        descriptors_path = os.path.join(args.dataset_path, "descriptors",
                                        image_name_ex + ".bin")
        if os.path.exists(keypoint_path) and \
                os.path.exists(descriptors_path):
            print(" -> skipping, already exist")
            continue

        start_time = time.time()

        image = imageio.imread(image_path)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
        resized_image = image
        if max(resized_image.shape) > args.max_edge:
            scaled_ratio = args.max_edge / max(resized_image.shape)
            scaled_size = (math.ceil(resized_image.shape[1]*scaled_ratio),
                           math.ceil(resized_image.shape[0]*scaled_ratio))
            resized_image = np.array(Image.fromarray(image).resize(scaled_size))\
                .astype(np.float32)
            # resized_image = scipy.misc.imresize(
            #     resized_image,
            #     args.max_edge / max(resized_image.shape)
            # ).astype('float')
        if sum(resized_image.shape[: 2]) > args.max_sum_edges:
            scaled_ratio = args.max_sum_edges / sum(resized_image.shape[: 2])
            scaled_size = (math.ceil(resized_image.shape[1] * scaled_ratio),
                           math.ceil(resized_image.shape[0] * scaled_ratio))
            resized_image = np.array(Image.fromarray(image).resize(scaled_size))\
                .astype(np.float32)
            # resized_image = scipy.misc.imresize(
            #     resized_image,
            #     args.max_sum_edges / sum(resized_image.shape[: 2])
            # ).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=args.preprocessing
        )
        with torch.no_grad():
            if args.multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    model,
                    scales=[1]
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]
        keypoints = np.column_stack((keypoints, np.ones((keypoints.shape[0], 1))))\
            .astype(np.float32)

        write_matrix(keypoint_path, keypoints)
        write_matrix(descriptors_path, descriptors)

        print(" in {:.3f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
