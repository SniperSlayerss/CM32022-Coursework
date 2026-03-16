import cv2 as cv
import numpy as np
from c1 import compare_histograms


def build_scaled_templates(template, template_scales):
    pyramid = {}

    base = template.copy()

    pyramid[1.0] = base

    # 1/2
    half = cv.pyrDown(base)
    pyramid[0.5] = half

    # 1/4
    quarter = cv.pyrDown(half)
    pyramid[0.25] = quarter

    # 1/8
    eighth = cv.pyrDown(quarter)
    pyramid[0.125] = eighth

    if 0.375 in template_scales:
        blurred = cv.GaussianBlur(base, (5, 5), 0)
        h, w = base.shape[:2]
        w_new = max(1, int(w * 0.375))
        h_new = max(1, int(h * 0.375))
        pyramid[0.375] = cv.resize(blurred, (w_new, h_new), interpolation=cv.INTER_CUBIC)

    return {s: pyramid[s] for s in template_scales if s in pyramid}


def epanechnikov_kernel(w, h):
    center_x, center_y = w // 2, h // 2
    y_indices, x_indices = np.indices((h, w))
    max_radius = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    grid = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2) / max_radius
    weights = np.where(grid <= 1, 3 / 4 * (1 - grid**2), 0)
    return weights.flatten()


def calculate_histogram(region, kernel):
    hist = np.bincount(region.ravel(), weights=kernel, minlength=256)
    norm_hist = cv.normalize(hist, None, alpha=0, beta=255, norm_type=cv.NORM_L1)
    return np.float32(norm_hist)


def template_matching(image, temps, temps_hist, kernels):
    step = 5
    search_size = 40
    alpha = 0.5
    boxes = []
    scores = []

    for t in range(temps.shape[0]):
        best_score = -1
        best_match = None
        best_temp = None

        count = 0
        for i in range(temps.shape[1]):
            template = temps[t][i]

            result_ncc = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

            _, _, _, max_loc = cv.minMaxLoc(result_ncc)

            x, y = max_loc
            h, w = template.shape[:2]

            search_x_start = max(0, x - search_size // 2)
            search_y_start = max(0, y - search_size // 2)
            search_x_end = min(image.shape[1] - w, x + search_size // 2)
            search_y_end = min(image.shape[0] - h, y + search_size // 2)

            ys = np.arange(search_y_start, search_y_end, step)
            xs = np.arange(search_x_start, search_x_end, step)
            coords = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

            # for check_y in range(search_y_start, search_y_end, step):
            #     for check_x in range(search_x_start, search_x_end, step):
            for check_x, check_y in coords:
                if check_y + h > image.shape[0] or check_x + w > image.shape[1]:
                    continue

                count += 1

                region = image[check_y : check_y + h, check_x : check_x + w]

                region_hist = calculate_histogram(region, kernels[t][i])
                hist_score = compare_histograms(temps_hist[t][i], region_hist)
                ncc_score = result_ncc[check_y, check_x]

                combined_score = alpha * ncc_score + (1 - alpha) * hist_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = (check_x, check_y)
                    best_temp = template

        h, w = best_temp.shape[:2]
        scores.append(best_score)
        boxes.append([best_match[0], best_match[1], best_match[0] + w, best_match[1] + h])

    return boxes, scores


def NMS(boxes, treshold=0):
    indices = np.arange(len(boxes))

    areas = (boxes[indices, 2] - boxes[indices, 0] + 1) * (boxes[indices, 3] - boxes[indices, 1] + 1)

    i = 0

    while i < indices.shape[0] - 1:
        temp_indices = indices[indices > i]

        xx1 = np.maximum(boxes[i][0], boxes[temp_indices, 0])
        yy1 = np.maximum(boxes[i][1], boxes[temp_indices, 1])
        xx2 = np.minimum(boxes[i][2], boxes[temp_indices, 2])
        yy2 = np.minimum(boxes[i][3], boxes[temp_indices, 3])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[temp_indices]

        indices = indices[~np.isin(indices, temp_indices[overlap > treshold])]

        i += 1

    return indices


def get_metrics(icon_dir, boxes, classidx, top, left, bottom, right, iou_threshold):
    tp = 0
    fp = 0
    fn = 0

    matched_gt = set()

    # compute IoU
    for i in range(len(boxes)):
        box1_left, box1_top, box1_right, box1_bottom = boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]

        pred_class = icon_dir[boxes[i][4]][:-4]

        if pred_class[0] == "0":
            pred_class = pred_class[1:]

        best_iou = 0
        best_gt = -1

        for j in range(len(classidx)):
            box2_left, box2_top, box2_right, box2_bottom = left[j], top[j], right[j], bottom[j]

            xx1 = np.maximum(box1_left, box2_left)
            yy1 = np.maximum(box1_top, box2_top)
            xx2 = np.minimum(box1_right, box2_right)
            yy2 = np.minimum(box1_bottom, box2_bottom)

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            box1_area = (box1_right - box1_left) * (box1_bottom - box1_top)
            box2_area = (box2_right - box2_left) * (box2_bottom - box2_top)

            intersection = w * h
            union = box1_area + box2_area - intersection

            if union > 0:
                iou = intersection / union
            else:
                iou = 0

            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold:
            if pred_class == classidx[best_gt]:
                tp += 1
            else:
                fp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(classidx) - len(matched_gt)

    return tp, fp, fn
