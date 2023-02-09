# Revised 2020-8-21: San Jiang <jiangsan@cug.edu.cn>
#  - Make this script adapt to the revision of the master COLMAP.
#  - Add some SOTA learned local feature descriptors.

import os
import argparse

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the dataset, e.g., path/to/Fountain")
    parser.add_argument("--matches_file", required=True,
                        help="Path to the matches file")
    parser.add_argument("--dump_path", required=True,
                        help="Path to the output path")
    parser.add_argument("--ratio", type=int, default=6,
                        help="Ratio value to scale the output file")
    parser.add_argument("--postfix", type=str, default="",
                        help="Postfix appeded to the output file")
    args = parser.parse_args()
    return args


def read_matrix(path, dtype):
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, count=2, dtype=np.int32)
        matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)


def main():
    args = parse_args()

    # Parse two image paths from the path of the given matches file.
    if not os.path.exists(args.matches_file):
        print(" -> skip, invalidate matches file")

    print("Draw matches for {}".format(args.matches_file))

    ratio = args.ratio
    image_path = os.path.join(args.dataset_path, "images")
    kpt_path = os.path.join(args.dataset_path, "keypoints")

    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    image_pair_name = os.path.splitext(os.path.basename(args.matches_file))[0]
    image_name1, image_name2 = image_pair_name.split('---')

    image_path1 = os.path.join(image_path, image_name1.replace('-', '/'))
    kpt_path1 = os.path.join(kpt_path, image_name1 + ".bin")
    image_path2 = os.path.join(image_path, image_name2.replace('-', '/'))
    kpt_path2 = os.path.join(kpt_path, image_name2 + ".bin")

    # Read the images, keypoints and matches.
    img1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
    h1, w1 = img1.shape[0:2]
    new_size1 = (int(w1/ratio), int(h1/ratio))
    img1 = cv2.resize(img1, new_size1)

    img2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)
    h2, w2 = img2.shape[0:2]
    new_size2 = (int(w2 / ratio), int(h2 / ratio))
    img2 = cv2.resize(img2, new_size2)

    matched_img = np.concatenate((img1, img2), axis=1)

    kpts1 = read_matrix(kpt_path1, np.float32)
    cv_kpts1 = [cv2.KeyPoint(p[0]/ratio, p[1]/ratio, p[2], p[3]) for p in kpts1]
    kpts2 = read_matrix(kpt_path2, np.float32)
    cv_kpts2 = [cv2.KeyPoint(p[0]/ratio, p[1]/ratio, p[2], p[3]) for p in kpts2]

    matches = read_matrix(args.matches_file, np.uint32)

    # If no matches exist.
    if matches.shape[0] == 0:
        if args.postfix.strip() == '':
            cv2.imwrite(os.path.join(args.dump_path, image_pair_name + \
                                     "-{}.JPG".format(matches.shape[0])), matched_img)
        else:
            cv2.imwrite(os.path.join(args.dump_path, image_pair_name + \
                                     "-{}-{}-{}.JPG".format(args.postfix, 0, 0)), matched_img)
        return 0

    # Determine inliers and outliers.
    good_kpts1 = np.array([cv_kpts1[m[0]].pt for m in matches])
    good_kpts2 = np.array([cv_kpts2[m[1]].pt for m in matches])

    err_thld = 1.0
    _, mask = cv2.findFundamentalMat(good_kpts1, good_kpts2, cv2.RANSAC, err_thld)

    n_all = mask.size
    n_inlier = np.count_nonzero(mask)

    # Draw matches and save the result.
    offset = new_size1[0]
    color_inlier = (0, 255, 0)
    color_outlier = (255, 0, 0)
    thickness = 2

    for idx in range(0, n_all):
        pt_start = (int(good_kpts1[idx][0]), int(good_kpts1[idx][1]))
        pt_end = (int(good_kpts2[idx][0] + offset), int(good_kpts2[idx][1]))
        if mask[idx]:
            cv2.line(matched_img, pt_start, pt_end, color_inlier, thickness)
        else:
            cv2.line(matched_img, pt_start, pt_end, color_outlier, thickness)

    # cv2.imshow("matched_img", matched_img)
    # cv2.waitKey(0)

    if args.postfix.strip() == '':
        cv2.imwrite(os.path.join(args.dump_path, image_pair_name + \
                                 "-{}.JPG".format(matches.shape[0])), matched_img)
    else:
        cv2.imwrite(os.path.join(args.dump_path, image_pair_name + \
                                 "-{}-{}-{}.JPG".format(args.postfix, n_inlier, n_all)), matched_img)


if __name__ == "__main__":
    main()