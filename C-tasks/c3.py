import csv
import itertools
import os
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import cv2 as cv
import numpy as np
import pandas as pd
from cv2.typing import MatLike

# Fix randomness
# np.random.seed(2424)

MIN_MATCH_COUNT = 35
MIN_INLIERS = 10
MIN_BOUNDING_BOX_AREA = 25
IOU_THRESHOLD = 0.85

NFEATURES = 4000
CONTRAST_THRESHOLD = 0.04

SCALES = [1.0, 0.7, 0.75, 0.25, 0.125, 0.1, 0.0625, 0.05, 0.025, 0.0125]

RANSAC_ITERATIONS = 1000
RANSAC_REPROJ_THRESH = 7.0
RANSAC_CONFIDENCE = 0.995


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


class Detection(NamedTuple):
    classname: str
    axis_aligned_box: np.ndarray
    rotated_quad: np.ndarray


@dataclass
class Detections:
    image: str
    detections: list[Detection]


# Returns good matches between the icon and image descriptor
def match_descriptors(
    icon_descriptor: np.ndarray,
    image_descriptor: np.ndarray,
    ratio: float,
) -> list[cv.DMatch]:
    # Compute the L2 distance

    icon2 = np.sum(icon_descriptor**2, axis=1, keepdims=True)
    img2 = np.sum(image_descriptor**2, axis=1, keepdims=True)

    dists = np.sqrt(icon2 + img2.T - 2.0 * (icon_descriptor @ image_descriptor.T))

    good_matches: list[cv.DMatch] = []

    for row_idx in range(icon_descriptor.shape[0]):
        row = dists[row_idx]

        # np.argpartition here will return array of indicies from row
        # such that the first k indecies correspond to the k smallest elements in row
        i1, i2 = np.argpartition(row, kth=2)[:2]
        if row[i1] > row[i2]:
            i1, i2 = i2, i1

        d1 = row[i1]
        d2 = row[i2]

        # Lowe ratio test
        if d1 < ratio * d2:
            good_matches.append(cv.DMatch(row_idx, i1, float(d1)))

    return good_matches


def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    dlt_matrix = []

    for (x_s, y_s), (x_d, y_d) in zip(src_pts, dst_pts):
        dlt_matrix.append([x_s, y_s, 1, 0, 0, 0, -x_d * x_s, -x_d * y_s, -x_d])
        dlt_matrix.append([0, 0, 0, x_s, y_s, 1, -y_d * x_s, -y_d * y_s, -y_d])

    dlt_matrix = np.array(dlt_matrix)

    _, _, V_t = np.linalg.svd(dlt_matrix)

    # The most right vector of V_t represents the vector with the smallest singular value (best homography solution)
    H = V_t[-1].reshape(3, 3)

    return H


def ransac_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    best_H: Optional[np.ndarray] = None
    best_inlier_count = 0
    best_mask: Optional[np.ndarray] = None

    if src_pts.shape[0] < 4:
        return None, None

    # Convert src_pts into homogenous coordinates
    src_pts_h = np.c_[src_pts, np.ones((src_pts.shape[0]))]

    iterations = RANSAC_ITERATIONS
    i = 0
    while i < iterations:
        i += 1

        # Randomly select four indexes/points
        idxs = np.random.choice(src_pts.shape[0], 4, replace=False)

        # Compute homographies
        H = compute_homography(src_pts[idxs], dst_pts[idxs])

        # Project all source points through H
        src_pts_h_proj = (H @ src_pts_h.T).T  # (N, 3)

        # Convert back to 2D
        src_pts_proj = src_pts_h_proj[:, :2] / (src_pts_h_proj[:, 2:3] + 1e-6)

        err = np.linalg.norm(src_pts_proj - dst_pts, axis=1)

        mask = err < RANSAC_REPROJ_THRESH
        inlier_count = int(mask.sum())

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_mask = mask
            best_H = H

            # Update the number of iterations we now need
            inlier_ratio = best_inlier_count / src_pts.shape[0]
            if inlier_ratio >= 1.0:
                break
            denom = np.log(max(1.0 - inlier_ratio**4, 1e-10))
            iterations = min(
                RANSAC_ITERATIONS,
                int(np.ceil(np.log(1.0 - RANSAC_CONFIDENCE) / denom)),
            )

    # Recompute the homography with the best mask
    best_H = compute_homography(src_pts[best_mask], dst_pts[best_mask])

    return best_H, best_mask


def parse_image(
    folder_name: str,
    file_name: str,
    is_test: bool = False,
    nfeatures: int = NFEATURES,
    n_octave_layers: int = 3,
) -> SIFTImage:
    image = cv.imread(folder_name + file_name)

    if image is None:
        print("image is none")
        exit(1)

    sift: cv.SIFT = cv.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=n_octave_layers,
        contrastThreshold=CONTRAST_THRESHOLD,
    )

    if is_test:
        kp, des = sift.detectAndCompute(image, None)
        return SIFTImage(file_name, image, kp, des)

    all_kp: list[cv.KeyPoint] = []
    all_des: list[np.ndarray] = []

    for scale in SCALES:
        resized = cv.resize(
            image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA
        )
        kp, des = sift.detectAndCompute(resized, None)
        if des is None:
            continue

        # Rescale keypoints back to original space
        for k in kp:
            k.pt = (k.pt[0] / scale, k.pt[1] / scale)
            k.size /= scale

        all_kp.extend(kp)
        all_des.append(des)

    combined = np.vstack(all_des) if all_des else None
    return SIFTImage(file_name, image, all_kp, combined)


def parse_annotation(folder_name: str, file_name: str) -> Detections:
    annotation = pd.read_csv(folder_name + file_name, encoding="ascii")

    real_detections = Detections(file_name, [])

    for _, row in annotation.iterrows():
        classname = row["classname"]
        left = float(row["left"])
        top = float(row["top"])
        right = float(row["right"])
        bottom = float(row["bottom"])
        bbox = np.array(
            [[[left, top]], [[left, bottom]], [[right, bottom]], [[right, top]]]
        )
        real_detections.detections.append(Detection(classname, bbox, bbox))

    return real_detections


def precompute_icons(
    icon_folder_name: str,
    nfeatures: int = NFEATURES,
    n_octave_layers: int = 3,
) -> list[SIFTImage]:
    data = []

    files = os.listdir(icon_folder_name)
    for icon in files:
        data.append(
            parse_image(
                icon_folder_name,
                icon,
                nfeatures=nfeatures,
                n_octave_layers=n_octave_layers,
            )
        )

    print("Parsed icons!")
    return data


def compute_metrics(
    real_detections: Detections, pred_detections: Detections
) -> tuple[int, int, int]:
    tp, fp = 0, 0
    matched_gt = set()

    for pred in pred_detections.detections:
        best_iou, best_idx = 0.0, -1

        for i, gt in enumerate(real_detections.detections):
            if i in matched_gt or gt.classname not in pred.classname:
                continue
            intersection, _ = cv.intersectConvexConvex(
                np.int32(pred.axis_aligned_box),
                np.int32(gt.axis_aligned_box),
                handleNested=True,
            )
            union = (
                cv.contourArea(np.int32(pred.axis_aligned_box))
                + cv.contourArea(np.int32(gt.axis_aligned_box))
                - intersection
            )
            iou = intersection / union
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_iou >= IOU_THRESHOLD and best_idx != -1:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(real_detections.detections) - len(matched_gt)
    return tp, fp, fn


def find_detections(
    image: SIFTImage,
    icon_descriptors: list[SIFTImage],
    save_matches: bool = False,
    img_number: int = 0,
    ratio: float = 0.75,
    min_match_count: int = MIN_MATCH_COUNT,
    min_inliers: int = MIN_INLIERS,
    min_bbox_area: int = MIN_BOUNDING_BOX_AREA,
) -> Detections:
    detections = Detections(image.name, [])
    match_vis_candidates: list[
        tuple[int, MatchResults, list[cv.DMatch], np.ndarray]
    ] = []

    results: list[MatchResults] = []
    for icon in icon_descriptors:
        good_matches = match_descriptors(icon.descriptor, image.descriptor, ratio)

        results.append(MatchResults(icon, image, good_matches))

    for res in results:
        if len(res.matches) < min_match_count:
            continue

        src_pts = np.float64(
            [res.query_image.keypoints[m.queryIdx].pt for m in res.matches]
        ).reshape(-1, 2)
        dst_pts = np.float64(
            [res.train_image.keypoints[m.trainIdx].pt for m in res.matches]
        ).reshape(-1, 2)

        M, mask = ransac_homography(src_pts, dst_pts)

        if M is None or mask is None:
            continue

        inliers = int(mask.ravel().sum())

        if inliers < min_inliers:
            continue

        if save_matches:
            match_vis_candidates.append((inliers, res, res.matches, mask))

        icon_h, icon_w, _ = res.query_image.image.shape

        pts = np.float32(
            [[0, 0], [0, icon_h - 1], [icon_w - 1, icon_h - 1], [icon_w - 1, 0]]
        ).reshape(-1, 1, 2)

        dst = cv.perspectiveTransform(pts, M)

        if not cv.isContourConvex(dst):
            continue

        if np.any(dst < 0):
            continue

        area = cv.contourArea(np.int32(dst))
        if area < min_bbox_area:
            continue

        # Check detected quad is within test image bounds
        xs = dst[:, 0, 0]
        ys = dst[:, 0, 1]

        left = int(xs.min())
        top = int(ys.min())
        right = int(xs.max())
        bottom = int(ys.max())

        # Tighten the axis_aligned_box more
        roi = image.image[top:bottom, left:right]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        # Blur the image so noise (small black pixels) are not included
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)
        coords = cv.findNonZero(thresh)

        x, y, w, h = cv.boundingRect(coords)
        left = left + x
        top = top + y
        right = left + w
        bottom = top + h

        axis_aligned_box = np.array(
            [[[left, top]], [[left, bottom]], [[right, bottom]], [[right, top]]]
        )

        rotated_quad = dst

        detections.detections.append(
            Detection(res.query_image.name, axis_aligned_box, rotated_quad)
        )

    # Save visualisations for the top 5 icon matches )
    match_vis_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, res, matches, mask in match_vis_candidates[:5]:
        save_match_visualisation(res, matches, mask, img_number)

    return detections


def save_match_visualisation(
    res: MatchResults,
    all_good_matches: list[cv.DMatch],
    inlier_mask: np.ndarray,
    img_number: int,
) -> None:
    output_dir = "output/"
    icon_img = res.query_image.image
    test_img = res.train_image.image
    icon_kp = res.query_image.keypoints
    test_kp = res.train_image.keypoints

    classname = res.query_image.name.replace(".png", "")

    before = cv.drawMatches(
        icon_img,
        icon_kp,
        test_img,
        test_kp,
        all_good_matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv.imwrite(f"{output_dir}matches_before_{img_number}_{classname}.jpg", before)

    # After RANSAC — inliers only
    inlier_matches = [
        m for m, keep in zip(all_good_matches, inlier_mask.ravel()) if keep
    ]
    after = cv.drawMatches(
        icon_img,
        icon_kp,
        test_img,
        test_kp,
        inlier_matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv.imwrite(f"{output_dir}matches_after_{img_number}_{classname}.jpg", after)


def draw_detections_on_image(
    image: MatLike,
    ground_truth_detections: Detections,
    predicted_detections: Detections,
) -> MatLike:
    output_image = image.copy()

    # Draw ground truth first (red axis-aligned boxes underneath)
    for det in ground_truth_detections.detections:
        cv.polylines(
            output_image,
            [np.int32(det.axis_aligned_box)],
            True,
            (0, 0, 255),
            3,
            cv.LINE_AA,
        )
        left = int(det.axis_aligned_box[0, 0, 0])
        top = int(det.axis_aligned_box[0, 0, 1])
        cv.putText(
            output_image,
            det.classname,
            (left, top - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

    # Draw predictions on top (green rotated quads)
    for det in predicted_detections.detections:
        cv.polylines(
            output_image,
            [np.int32(det.rotated_quad)],
            True,
            (0, 255, 0),
            3,
            cv.LINE_AA,
        )
        lx = int(det.rotated_quad[0, 0, 0])
        ly = int(det.rotated_quad[0, 0, 1])
        cv.putText(
            output_image,
            det.classname,
            (lx, ly - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

    return output_image


def grid_search(icon_folder_name: str, test_folder_name: str) -> None:
    images_folder = os.path.join(test_folder_name, "images/")
    annotations_folder = os.path.join(test_folder_name, "annotations/")

    nfeatures_vals = [1000, 2000, 4000]
    n_octave_vals = [3, 4, 6]
    ratio_vals = [0.60, 0.65, 0.70, 0.75]
    min_match_vals = [10, 20, 35]
    min_inlier_vals = [6, 10, 20]
    min_bbox_vals = [25, 100]

    inner_combos = list(
        itertools.product(ratio_vals, min_match_vals, min_inlier_vals, min_bbox_vals)
    )
    total = len(nfeatures_vals) * len(n_octave_vals) * len(inner_combos)

    ground_truths = [
        parse_annotation(annotations_folder, f"test_image_{i}.csv")
        for i in range(1, 21)
    ]

    results = []
    combo_idx = 0

    for nfeat, n_oct in itertools.product(nfeatures_vals, n_octave_vals):
        icon_descriptors = precompute_icons(icon_folder_name, nfeat, n_oct)
        test_images = [
            parse_image(
                images_folder,
                f"test_image_{i}.png",
                is_test=True,
                nfeatures=nfeat,
                n_octave_layers=n_oct,
            )
            for i in range(1, 21)
        ]

        for ratio, min_match, min_inlier, min_bbox in inner_combos:
            combo_idx += 1
            t0 = time.perf_counter()

            total_tp = total_fp = total_fn = 0
            for img, gt in zip(test_images, ground_truths):
                dets = find_detections(
                    img,
                    icon_descriptors,
                    ratio=ratio,
                    min_match_count=min_match,
                    min_inliers=min_inlier,
                    min_bbox_area=min_bbox,
                )
                tp, fp, fn = compute_metrics(gt, dets)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            elapsed = time.perf_counter() - t0
            total_preds = total_tp + total_fp + total_fn
            acc = total_tp / total_preds if total_preds > 0 else 0.0
            tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            fpr = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0.0

            print(
                f"[{combo_idx:3}/{total}] nfeat={nfeat:4}, oct={n_oct}, "
                f"ratio={ratio:.2f}, min_match={min_match:2}, "
                f"inliers={min_inlier:2}, bbox={min_bbox:3} "
                f"-> ACC={acc:.4f}  TP={total_tp:3} FP={total_fp:3} FN={total_fn:3}  t={elapsed:.1f}s"
            )

            results.append(
                {
                    "nfeatures": nfeat,
                    "n_octave_layers": n_oct,
                    "ratio_threshold": ratio,
                    "min_match_count": min_match,
                    "min_inliers": min_inlier,
                    "min_bbox_area": min_bbox,
                    "accuracy": acc,
                    "tpr": tpr,
                    "fpr": fpr,
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                    "runtime": elapsed,
                }
            )

    results.sort(key=lambda r: r["accuracy"], reverse=True)
    with open("grid_search_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nTop 5 results:")
    for r in results[:5]:
        print(
            f"  ACC={r['accuracy']:.4f}  TPR={r['tpr']:.3f}  FPR={r['fpr']:.3f}  "
            f"nfeat={r['nfeatures']} oct={r['n_octave_layers']} "
            f"ratio={r['ratio_threshold']} min_match={r['min_match_count']} "
            f"inliers={r['min_inliers']} bbox={r['min_bbox_area']}"
        )
    print("\nResults saved to grid_search_results.csv")
