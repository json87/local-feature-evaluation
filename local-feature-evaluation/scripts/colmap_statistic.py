# Revised 2020-11-12: San Jiang <jiangsan@cug.edu.cn>
#  - Make this script adapt to the revision of the master COLMAP.
#  - Export the statistic results of image matching.

import os
import argparse
import sqlite3
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--postfix", type=str, default="",
                        help="Postfix appeded to the output file")
    args = parser.parse_args()
    return args


def pair_id_to_image_ids(pair_id):
    image_id2 = int(pair_id % 2147483647)
    image_id1 = int((pair_id - image_id2) / 2147483647)
    return (image_id1, image_id2)


def main():
    args = parse_args()

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    image_overlap = {}
    cursor.execute("SELECT image_id FROM images;")
    for row in cursor:
        image_overlap[row[0]] = 0

    pair_ids = []
    cursor.execute("SELECT pair_id FROM two_view_geometries;")
    pair_ids = list(row for row in cursor)

    image_matches = []
    for pair_id, in pair_ids:
        cursor.execute("SELECT rows FROM matches WHERE pair_id=?;",
                       (pair_id,))
        initial_matches = next(cursor)[0]

        cursor.execute("SELECT rows FROM two_view_geometries WHERE pair_id=?;",
                       (pair_id,))
        inlier_matches = next(cursor)[0]

        if inlier_matches > 0:
            image_matches.append((pair_id, initial_matches, inlier_matches))
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_overlap[image_id1] = image_overlap[image_id1] + 1
            image_overlap[image_id2] = image_overlap[image_id2] + 1

    image_matches = np.array(image_matches)
    inlier_ratios = image_matches[:,2] / image_matches[:,1]
    inlier_mean = np.mean(inlier_ratios)
    inlier_std = np.std(inlier_ratios)

    # write the statistic results.
    if args.postfix.strip() == '':
        statistic_inlier_path = os.path.join(args.output_path, "statistic_inlier.txt")
        statistic_overlap_path = os.path.join(args.output_path, "statistic_overlap.txt")
    else:
        statistic_inlier_path = os.path.join(args.output_path, "statistic_inlier-{}.txt").format(args.postfix)
        statistic_overlap_path = os.path.join(args.output_path, "statistic_overlap-{}.txt").format(args.postfix)

    with open(statistic_inlier_path, "w") as fid:
        fid.write("{:.3f} {:.3f}\n".format(inlier_mean, inlier_std))
        for pair_id, initial_matches, inlier_matches in image_matches:
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            fid.write("{} {} {} {} {:.3f}\n".format(image_id1, image_id2,
                                             initial_matches, inlier_matches, inlier_matches/initial_matches))

    with open(statistic_overlap_path, "w") as fid:
        for image_id in image_overlap:
            fid.write("{} {}\n".format(image_id, image_overlap[image_id]))


if __name__ == "__main__":
    main()