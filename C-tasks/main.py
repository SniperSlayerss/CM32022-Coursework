import argparse
import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
from c1 import Canny, Hough_Transform, compute_angle
from c2 import NMS, build_scaled_templates, calculate_histogram, epanechnikov_kernel, get_metrics, template_matching
from c3 import (
    compute_metrics,
    draw_detections_on_image,
    find_detections,
    grid_search,
    parse_annotation,
    parse_image,
    precompute_icons,
)


def draw_lines(theta, r, image, f_name):
    # polar to cartesian coordinates
    a, b = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))

    x0, y0 = r * a, r * b
    x1 = (x0 + 1000 * (-b)).astype(int)
    y1 = (y0 + 1000 * (a)).astype(int)
    x2 = (x0 - 1000 * (-b)).astype(int)
    y2 = (y0 - 1000 * (a)).astype(int)

    for l in range(len(theta)):
        cv.line(image, (x1[l], y1[l]), (x2[l], y2[l]), (255, 0, 0), 1)

    cv.imshow(f"{f_name}", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def draw_box(folder, file, image, boxes):
    for b in boxes:
        cv.rectangle(image, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
        cv.putText(
            image,
            folder[b[4]],
            (b[0], b[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

        print(folder[b[4]])

    cv.imshow(f"{file}", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_task_c1(folder_name):
    # assume that this folder name has a file list.txt that contains the annotation.

    task1_data = pd.read_csv(folder_name + "/list.txt")
    # Write code to read in each image
    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image

    total_error = 0

    for f, a in zip(task1_data["FileName"], task1_data["AngleInDegrees"]):
        image = cv.imread(f"{folder_name}/" + f, cv.IMREAD_GRAYSCALE)

        if image is None:
            print("Error: Image not found or unable to read.")
            return

        # Write code to process the image
        height, width = image.shape
        dia_len = int(np.sqrt(height**2 + width**2))

        canny = Canny(image, 80, 200)

        theta, r = Hough_Transform(canny, dia_len)

        angle = compute_angle(theta)

        total_error += np.abs(int(a) - angle)

    print(total_error)
    return total_error


def test_task_c2(icon_dir, test_dir):
    # give input as icon_dir/png/   and      test_dir

    image_dir = test_dir + "/images/"
    annotate_dir = test_dir + "/annotations/"
    icon_folder = os.listdir(icon_dir)

    if not os.path.exists("predictions.npy"):
        templates = []

        for file in icon_folder:
            template = cv.imread(icon_dir + file, cv.IMREAD_GRAYSCALE)
            templates.append(template)

        pred = {}

        template_scales = [0.125, 0.25, 0.375, 0.5, 1]

        for file in os.listdir(image_dir):
            test_image = cv.imread(f"{image_dir}{file}")

            if test_image is None:
                print("Error: Image not found or unable to read.")
                return

            gray = cv.cvtColor(test_image, cv.COLOR_RGB2GRAY)

            multi_templates = np.zeros([len(templates), len(template_scales)], dtype=object)
            templates_kernel = np.zeros([len(templates), len(template_scales)], dtype=object)
            templates_hist = np.zeros([len(templates), len(template_scales)], dtype=object)

            for t in range(len(templates)):
                template = templates[t]

                scaled_templates = build_scaled_templates(template, template_scales)

                for s in range(len(template_scales)):
                    # compute kernel and template intensity histogram
                    # temp = cv.resize(template , None, fx = template_scales[s], fy = template_scales[s], interpolation = cv.INTER_CUBIC)

                    temp = scaled_templates[template_scales[s]]

                    kernel = epanechnikov_kernel(temp.shape[1], temp.shape[0])
                    temp_hist = calculate_histogram(temp, kernel)

                    multi_templates[t][s] = temp
                    templates_kernel[t][s] = kernel
                    templates_hist[t][s] = temp_hist

            # template matching
            boxes, scores = template_matching(gray, multi_templates, templates_hist, templates_kernel)
            arr = np.column_stack((np.array(boxes), np.arange(len(boxes)), np.array(scores)))
            set = arr[arr[:, 5].argsort()[::-1]]

            indices = NMS(set[:, :4])
            boxes = set[indices].astype(np.int32)

            pred[file[:-4]] = boxes

            print(f"complete: {file}")

            # draw_box(icon_folder, file, test_image, boxes)

        # Save predictions (Model predictions are saved here)
        np.save("predictions.npy", pred, allow_pickle=True)

    else:
        # Load model predictions if file exists in parent directory
        pred = np.load("predictions.npy", allow_pickle=True).item()

    # Compute IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives

    t_tp = 0
    t_fp = 0
    t_fn = 0

    for f in os.listdir(annotate_dir):
        annotate_f = pd.read_csv(f"{annotate_dir}{f}")
        classidx = annotate_f["classname"].tolist()
        top = annotate_f["top"].tolist()
        left = annotate_f["left"].tolist()
        bottom = annotate_f["bottom"].tolist()
        right = annotate_f["right"].tolist()

        boxes = pred[f[:-4]]

        tp, fp, fn = get_metrics(icon_folder, boxes, classidx, top, left, bottom, right, iou_threshold=0.85)

        t_tp += tp
        t_fp += fp
        t_fn += fn

    acc = t_tp / (t_tp + t_fp + t_fn + 1e-6)
    tpr = t_tp / (t_tp + t_fn + 1e-6)
    fpr = t_fp / (t_fp + t_tp + 1e-6)
    fnr = t_fn / (t_tp + t_fn + 1e-6)

    # acc, tpr, fpr, fnr = None, None, None, None
    print(acc, tpr, fpr, fnr)
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
        test_image = parse_image(image_dir, f"test_image_{img_number}.png", is_test=True)

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

        real_detections = parse_annotation(annotations_dir, f"test_image_{img_number}.csv")

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
    acc = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

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
