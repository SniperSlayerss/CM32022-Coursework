import argparse
import os
import random
from dataclasses import dataclass
from typing import Sequence

import cv2 as cv
import numpy as np
import pandas as pd
from cv2.typing import MatLike

# MIN_MATCH_COUNT = 20
MIN_INLIERS = 6
MIN_BOUNDING_BOX_AREA = 40

FLANN_INDEX_KDTREE = 1


@dataclass
class SIFTImage:
    name: str
    image: MatLike
    keypoints: Sequence[cv.KeyPoint]
    descriptor: MatLike


@dataclass
class MatchResults:
    query_image: SIFTImage
    train_image: SIFTImage
    matches: list[cv.DMatch]


def test_task_c1(folder_name):
    # assume that this folder name has a file list.txt that contains the annotation.
    task1_data = pd.read_csv(folder_name + "/list.txt")
    # task1_data = pd.read_csv(folder_name + "/list.txt")
    # Write code to read in each image
    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image
    total_error = None
    return total_error


def test_task_c2(icon_dir, test_dir):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    acc, tpr, fpr, fnr = None, None, None, None
    return acc, tpr, fpr, fnr


def parse_image(folder_name, file_name) -> SIFTImage:
    image = cv.imread(folder_name + file_name)
    if image is None:
        print("image is none")
        exit(1)

    # image = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

    sift: cv.SIFT = cv.SIFT_create(
        nfeatures=2000,
    )

    kp, des = sift.detectAndCompute(image, None)

    return SIFTImage(file_name, image, kp, des)


def precompute_icons(icon_folder_name) -> list[SIFTImage]:
    data = []

    files = os.listdir(icon_folder_name)
    for icon in files:
        data.append(parse_image(icon_folder_name, icon))

    print("Parsed icons!")
    return data


def test_task_c3(icon_folder_name, test_folder_name):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives

    # img = cv.imread(icon_dir + "001-lighthouse.png")

    icon_descriptors = precompute_icons(icon_folder_name)

    img_number = random.randint(1, 20)

    test_image = parse_image(test_folder_name, f"test_image_{img_number}.png")

    print(f"Test Image: {test_image.name}")

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    results: list[MatchResults] = []
    for icon in icon_descriptors:
        matches = flann.knnMatch(icon.descriptor, test_image.descriptor, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        results.append(MatchResults(icon, test_image, good_matches))

    output_image = test_image.image

    results.sort(key=lambda x: len(x.matches), reverse=True)

    results = results[:8]

    for res in results:
        matches = res.matches

        if len(res.matches) < 4:
            continue

        print(res.query_image.name, len(res.matches))
        # if len(res.matches) < MIN_MATCH_COUNT:
        #    continue

        src_pts = np.float32(
            [res.query_image.keypoints[m.queryIdx].pt for m in res.matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [res.train_image.keypoints[m.trainIdx].pt for m in res.matches]
        ).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if M is None:
            continue

        inliers = sum(mask.ravel().tolist())

        if inliers < MIN_INLIERS:
            continue

        h, w, _ = res.query_image.image.shape

        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )

        dst = cv.perspectiveTransform(pts, M)

        if cv.contourArea(np.int32(dst)) < MIN_BOUNDING_BOX_AREA:
            continue

        # draw rotated bounding box
        cv.polylines(
            output_image,
            [np.int32(dst)],
            True,
            (0, 255, 0),
            3,
            cv.LINE_AA,
        )
        # axis-aligned bbox (for evaluation)
        xs = dst[:, 0, 0]
        ys = dst[:, 0, 1]

        left = max(0, int(xs.min()))
        top = max(0, int(ys.min()))
        right = min(test_image.image.shape[1], int(xs.max()))
        bottom = min(test_image.image.shape[0], int(ys.max()))

        # label
        cv.putText(
            output_image,
            res.query_image.name,
            (left, top - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        cv.imwrite("output.jpg", output_image)

    debug = cv.drawKeypoints(
        test_image.image,
        test_image.keypoints,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    cv.imwrite("kp_debug.jpg", debug)


if __name__ == "__main__":
    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument(
        "--Task1Dataset",
        help="Provide a folder that contains the Task 1 Dataset.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--IconDataset",
        help="Provide a folder that contains the Icon Dataset for Task2 and Task3.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--Task2Dataset",
        help="Provide a folder that contains the Task 2 test Dataset.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--Task3Dataset",
        help="Provide a folder that contains the Task 3 test Dataset.",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    if args.Task1Dataset is not None:
        # This dataset has a list of png files and a txt file that has annotations of filenames and angle
        test_task_c1(args.Task1Dataset)

    if args.IconDataset is not None and args.Task2Dataset is not None:
        # The Icon dataset has a directory that contains the icon image for each file
        # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a
        # png directory with list of images
        test_task_c2(args.IconDataset, args.Task2Dataset)

    if args.IconDataset is not None and args.Task3Dataset is not None:
        # The Icon dataset directory contains an icon image for each file
        # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png
        # directory with list of images
        test_task_c3(args.IconDataset, args.Task3Dataset)
