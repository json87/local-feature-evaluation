# Revised 2020-7-27: San Jiang <jiangsan@cug.edu.cn>
#  - Make this script adapt to the revision of the master COLMAP.
#  - Add some SOTA learned local feature descriptors.
#
# contextdesc - https://github.com/lzx551402/contextdesc

import os
import cv2
import time
import argparse
import numpy as np
import h5py

from queue import Queue
from threading import Thread

import tensorflow as tf

from utils.tf import recoverer
from models import get_model
from models.inference_model import inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the dataset, e.g., path/to/Fountain")
    parser.add_argument("--model_path", required=True,
                        help="Path to the pretained contextdesc model")
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


def read_matrix(path, dtype):
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, count=2, dtype=np.int32)
        matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)


def write_matrix(path, matrix):
    with open(path, "wb") as fid:
        shape = np.array(matrix.shape, dtype=np.int32)
        shape.tofile(fid)
        matrix.tofile(fid)


def prepare_regional_features(dataset_path, image_names, reg_model):
    in_img_name = []
    in_img_path = []
    out_img_feat_list = []
    for image_name in image_names:
        seperator_pos = len(os.path.join(dataset_path,
                                         "images").replace('\\', '/')) + 1
        image_name_ex = image_name[seperator_pos:].replace('/', '-')
        image_path = os.path.join(dataset_path, image_name)
        reg_feat_path = os.path.join(dataset_path, "descriptors",
                                     image_name_ex + ".bin.feat.npy")
        if not os.path.exists(reg_feat_path):
            in_img_name.append(image_name)
            in_img_path.append(image_path)
            out_img_feat_list.append(reg_feat_path)

    if len(in_img_path) > 0:
        model = get_model('reg_model')(reg_model)
        for idx, image_path in enumerate(in_img_path):
            print('Compute regional features for {} [{}/{}]'.format(
                in_img_name[idx], idx+1, len(in_img_path)), end='')

            start_time = time.time()

            img = cv2.imread(image_path)
            img = img[..., ::-1]
            reg_feat = model.run_test_data(img)
            np.save(out_img_feat_list[idx], reg_feat)

            print(" in {:.3f}s".format(time.time() - start_time))
        model.close()


def loader(dataset_path, image_names, producer_queue):
    for idx, image_name in enumerate(image_names):
        seperator_pos = len(os.path.join(dataset_path,
                                         "images").replace('\\', '/')) + 1
        image_name_ex = image_name[seperator_pos:].replace('/', '-')
        image_path = os.path.join(dataset_path, image_name)
        keypoint_path = os.path.join(dataset_path, "keypoints",
                                     image_name_ex + ".bin")
        patches_path = os.path.join(dataset_path, "descriptors",
                                    image_name_ex + ".bin.patches.mat")
        img_feat_path = os.path.join(dataset_path, "descriptors",
                                     image_name_ex + ".bin.feat.npy")

        if not os.path.exists(image_path) or \
            not os.path.exists(keypoint_path) or \
                not os.path.exists(patches_path) or \
                not os.path.exists(img_feat_path):
            continue

        descriptors_path = os.path.join(dataset_path, "descriptors",
                                        image_name_ex + ".bin")

        if os.path.exists(descriptors_path):
            print("Computing features for {} [{}/{}]".format(
                image_name, idx + 1, len(image_names)), end="")
            print(" -> skipping, already exist")
            continue


        # read image features.
        img_feat = np.load(img_feat_path)

        # read image patches.
        with h5py.File(patches_path, 'r') as patches_file:
            patches31 = np.array(patches_file["patches"]).T

        patches = np.empty((patches31.shape[0], 32, 32), dtype=np.float32)
        patches[:, :31, :31] = patches31
        patches[:, 31, :31] = patches31[:, 30, :]
        patches[:, :31, 31] = patches31[:, :, 30]
        patches[:, 31, 31] = patches31[:, 30, 30]

        # read keypoint parameters.
        img_size = cv2.imread(image_path).shape
        keypoints = read_matrix(keypoint_path, np.float32)
        kpt_num = keypoints.shape[0]
        kpt_param = np.zeros((kpt_num, 6))
        kpt_param[:, 2] = (keypoints[:, 0] - img_size[1] / 2.) / (img_size[1] / 2.)
        kpt_param[:, 5] = (keypoints[:, 1] - img_size[0] / 2.) / (img_size[0] / 2.)

        producer_queue.put([idx, len(image_names), image_name, descriptors_path,
                            kpt_param, patches, img_feat])

    producer_queue.put(None)


def extractor(patch_queue, sess, output_tensors, batch_size, consumer_queue):
    while True:
        queue_data = patch_queue.get()
        if queue_data is None:
            consumer_queue.put(None)
            return
        idx, image_number, image_name, descriptors_path, kpt_param, patches, img_feat = queue_data

        print("Computing features for {} [{}/{}]".format(
            image_name, idx + 1, image_number), end="")

        start_time = time.time()

        descriptors = []
        for i in range(0, patches.shape[0], batch_size):
            patches_batch = \
                patches[i:min(i + batch_size, patches.shape[0])]
            kpt_param_batch = \
                kpt_param[i:min(i + batch_size, patches.shape[0])]

            input_dict = {'ph_patch:0': patches_batch, 'ph_kpt_param:0': np.expand_dims(kpt_param_batch, axis=0),
                          'ph_img_feat:0': np.expand_dims(img_feat, axis=0)}
            output_arrays = sess.run(output_tensors, input_dict)
            descriptors.append(output_arrays['local_feat'].reshape(-1, 128))

        if len(descriptors) == 0:
            descriptors = np.zeros((0, 128), dtype=np.float32)
        else:
            descriptors = np.concatenate(descriptors)
        consumer_queue.put([descriptors_path, descriptors])

        print(" in {:.3f}s".format(time.time() - start_time))

        patch_queue.task_done()


def writer(consumer_queue):
    while True:
        queue_data = consumer_queue.get()
        if queue_data is None:
            return
        descriptors_path, descriptors = queue_data
        write_matrix(descriptors_path, descriptors)


def main():
    args = parse_args()

    # Specify the paths of the pre-trained model.
    reg_model_path = os.path.join(args.model_path, 'reg.pb')
    loc_model_path = os.path.join(args.model_path, 'model.ckpt-400000')

    if not os.path.exists(os.path.join(args.dataset_path, "descriptors")):
        os.makedirs(os.path.join(args.dataset_path, "descriptors"))

    # image_names = os.listdir(os.path.join(args.dataset_path, "images"))
    image_names = []
    list_dir_recursive(os.path.join(args.dataset_path, "images"), image_names)

    # Prepare regional features.
    prepare_regional_features(args.dataset_path, image_names, reg_model_path)

    # Construct inference networks.
    output_tensors = inference({'dense_desc': False,
                                'reg_feat_dim': 2048,
                                'aug': True})

    # Create the initializier.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        # Restore pre-trained model.
        recoverer(sess, loc_model_path)

        producer_queue = Queue(maxsize=18)
        consumer_queue = Queue()

        producer0 = Thread(target=loader, args=(
            args.dataset_path, image_names, producer_queue))
        producer0.daemon = True
        producer0.start()

        producer1 = Thread(target=extractor, args=(
            producer_queue, sess, output_tensors, args.batch_size, consumer_queue))
        producer1.daemon = True
        producer1.start()

        consumer = Thread(target=writer, args=(consumer_queue,))
        consumer.daemon = True
        consumer.start()

        producer0.join()
        producer1.join()
        consumer.join()


if __name__ == "__main__":
    main()
