# Import the features and matches into a COLMAP database.
#
# Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

# Revised 2020-7-17: San Jiang <jiangsan@cug.edu.cn>
#  Make this script adapt to the revision of the master COLMAP.

from __future__ import print_function, division

import os
import sys
import glob
import argparse
import sqlite3
import subprocess
import multiprocessing

import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--colmap_path", required=True,
                        help="Path to the COLMAP executable folder, e.g., "
                             "path/to/colmap/build/src/exe")
    args = parser.parse_args()
    return args


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def read_matrix(path, dtype):
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, count=2, dtype=np.int32)
        matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)


def main():
    args = parse_args()

    connection = sqlite3.connect(os.path.join(args.dataset_path, "database.db"))
    cursor = connection.cursor()

    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    cursor.execute("DELETE FROM two_view_geometries;")
    connection.commit()

    # import keypoints from external files.

    images = {}
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]

    for image_name, image_id in images.items():
        print("Importing features for", image_name)
        image_name_ex = image_name.replace('/', '-')
        keypoint_path = os.path.join(args.dataset_path, "keypoints",
                                     image_name_ex + ".bin")
        keypoints = read_matrix(keypoint_path, np.float32)
        # add by san jiang. Notice that the origin of the pixel coordinate
        # is located at the corner of the top-left pixel for colmap. This
        # offset is consistent with that in line 62 of colmap_export.py.
        keypoints[:, :2] += 0.5
        # descriptor_path = os.path.join(args.dataset_path, "descriptors",
        #                              image_name_ex + ".bin")
        # descriptors = read_matrix(descriptor_path, np.float32)
        # assert keypoints.shape[1] == 4
        # assert keypoints.shape[0] == descriptors.shape[0]
        if IS_PYTHON3:
            keypoints_str = keypoints.tostring()
        else:
            keypoints_str = np.getbuffer(keypoints)
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) "
                       "VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1],
                        keypoints_str))
        connection.commit()

    # import matches from external files.

    image_pairs = []
    for match_path in glob.glob(os.path.join(args.dataset_path,
                                             "matches/*---*.bin")):
        image_name1_ex, image_name2_ex = \
            os.path.basename(match_path[:-4]).split("---")
        image_name1 = image_name1_ex.replace('-', '/')
        image_name2 = image_name2_ex.replace('-', '/')
        image_pairs.append((image_name1, image_name2))
        print("Importing matches for", image_name1, "---", image_name2)
        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = read_matrix(match_path, np.uint32)
        assert matches.shape[1] == 2
        if IS_PYTHON3:
            matches_str = matches.tostring()
        else:
            matches_str = np.getbuffer(matches)
        cursor.execute("INSERT INTO  matches(pair_id, rows, cols, data) "
                       "VALUES(?, ?, ?, ?);",
                       (image_pair_id, matches.shape[0], matches.shape[1],
                        matches_str))
        connection.commit()

    with open(os.path.join(args.dataset_path, "image-pairs.txt"), "w") as fid:
        for image_name1, image_name2 in image_pairs:
            fid.write("{} {}\n".format(image_name1, image_name2))

    cursor.close()
    connection.close()

    # execute geometric verification.

    subprocess.call([os.path.join(args.colmap_path, "colmap"),
                     "matches_importer",
                     "--database_path",
                     os.path.join(args.dataset_path, "database.db"),
                     "--match_list_path",
                     os.path.join(args.dataset_path, "image-pairs.txt"),
                     "--match_type", "pairs",
                     "--SiftMatching.max_error", "1.0"])


if __name__ == "__main__":
    main()
