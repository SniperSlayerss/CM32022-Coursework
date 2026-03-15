import argparse
import os
import time

import cv2 as cv
import pandas as pd
from c3 import (
    compute_metrics,
    draw_detections_on_image,
    find_detections,
    grid_search,
    parse_annotation,
    parse_image,
    precompute_icons,
)


def test_task_c1(folder_name):
    # assume that this folder name has a file list.txt that contains the annotation.
    task1_data = pd.read_csv(folder_name + "/list.txt")
    # Write code to read in each image
    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image
    total_error = None
    return total_error


def test_task_c2(icon_dir, test_dir):
    images_folder = os.path.join(test_dir, "images/")
    annotations_folder = os.path.join(test_dir, "annotations/")
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    acc, tpr, fpr, fnr = None, None, None, None
    return acc, tpr, fpr, fnr


def test_task_c3(icon_folder_name, test_folder_name):
    image_dir = f"{test_folder_name}/images/"
    annotations_dir = f"{test_folder_name}annotations/"

    # Precompute descriptors for all icons
    icon_descriptors = precompute_icons(icon_folder_name)

    total_tp, total_fp, total_fn = 0, 0, 0
    runtimes = []

    # For these images, we draw the matches in an image
    draw_matches_images = [1, 2]

    for img_number in range(1, 21):
        test_image = parse_image(
            image_dir, f"test_image_{img_number}.png", is_test=True
        )

        draw_match = img_number in draw_matches_images

        start_time = time.perf_counter()

        detections = find_detections(
            test_image,
            icon_descriptors,
            save_matches=draw_match,
            img_number=img_number,
        )
        end_time = time.perf_counter()

        runtimes.append(end_time - start_time)

        real_detections = parse_annotation(
            annotations_dir, f"test_image_{img_number}.csv"
        )

        output_image = draw_detections_on_image(
            test_image.image,
            ground_truth_detections=real_detections,
            predicted_detections=detections,
        )

        cv.imwrite(f"output/output_image_{img_number}.jpg", output_image)

        tp, fp, fn = compute_metrics(real_detections, detections)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"Parsed {test_image.name}!")

    tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    fpr = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0.0
    acc = (
        total_tp / (total_tp + total_fp + total_fn)
        if (total_tp + total_fp + total_fn) > 0
        else 0.0
    )

    avg_runtime = sum(runtimes) / len(runtimes)
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"ACC: {acc:.3f}, TPR: {tpr:.3f}, FPR: {fpr:.3f}")
    print(f"Avg runtime per image: {avg_runtime:.3f}s")


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
    parser.add_argument(
        "--GridSearch",
        help="Run grid search over hyperparameters. Provide the Task 3 Dataset folder.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--ransac",
        help=(
            "RANSAC implementation to use for Task 3. "
            "'custom' (default) uses the manual DLT+RANSAC required for graded submission. "
            "'opencv' uses cv2.findHomography — DEV/DEBUG ONLY, forbidden by the coursework spec."
        ),
        choices=["custom", "opencv"],
        default="custom",
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
        test_task_c3(
            args.IconDataset,
            args.Task3Dataset,
        )

    if args.IconDataset is not None and args.GridSearch is not None:
        grid_search(args.IconDataset, args.GridSearch)
