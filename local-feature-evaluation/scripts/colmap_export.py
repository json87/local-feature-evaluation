# Export the keypoints of a COLMAP database to a directory.
#
# Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

# Revised 2020-7-17: San Jiang <jiangsan@cug.edu.cn>
#  Make this script adapt to the revision of the master COLMAP.

import os
import argparse
import sqlite3
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    try:
        os.makedirs(args.output_path)
    except:
        pass

    # export cameras.

    cameras = {}
    cursor.execute("SELECT camera_id, params FROM cameras;")
    for row in cursor:
        camera_id = row[0]
        params = np.fromstring(row[1], dtype=np.double)
        cameras[camera_id] = params

    # export keypoints to external files.

    images = []
    cursor.execute("SELECT image_id, name FROM images;")
    images = list(row for row in cursor)

    for image_id, image_name in images:
        base_name, ext = os.path.splitext(image_name)
        image_name = image_name.replace('/', '-')
        image_name = image_name.replace('\\', '-')
        file_name = os.path.join(args.output_path, image_name + ".bin")
        if os.path.exists(file_name):
            continue

        print("Exporting keypoints for", image_name)

        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;",
                       (image_id,))
        row = next(cursor)
        if row[0] is None:
            keypoints = np.zeros((0, 4), dtype=np.float32)
        else:
            keypoints = np.fromstring(row[0], dtype=np.float32).reshape(-1, 6)

            keypoints[:, :2] -= 0.5

            scale_x = (keypoints[:, 2]**2 + keypoints[:, 4]**2)**0.5;
            scale_y = (keypoints[:, 3]**2 + keypoints[:, 5]**2)**0.5;
            scale = (scale_x + scale_y) / 2.0;
            orientation = np.arctan2(keypoints[:, 4], keypoints[:, 2]);

            keypoints[:, 2] = scale;
            keypoints[:, 3] = orientation;

            keypoints = np.delete(keypoints, [4, 5], 1);

        with open(file_name, "wb") as fid:
            shape = np.array(keypoints.shape, dtype=np.int32)
            shape.tofile(fid)
            keypoints.tofile(fid)

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
