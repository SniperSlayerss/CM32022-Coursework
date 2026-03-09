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


def Canny(image, h1=50, h2=100):

    gaussian = cv.GaussianBlur(image, (5,5), 0)

    # Sobel Operator - Image gradient
    sobelx = cv.Sobel(gaussian,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(gaussian,cv.CV_64F,0,1,ksize=5)

    # Gradient Magitude
    grad_magnitude = np.sqrt(sobelx**2 + sobely**2)
    grad_magnitude = cv.normalize(grad_magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Gradient Orientation
    grad_orientation = np.rad2deg(np.arctan2(sobely,sobelx))
    grad_orientation[grad_orientation < 0] += 180

    h, w = image.shape
    points = np.zeros((h,w))
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            angle = grad_orientation[i][j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = grad_magnitude[i,j+1], grad_magnitude[i,j-1]
            elif 22.5 <= angle < 67.5:
                q, r = grad_magnitude[i+1,j-1], grad_magnitude[i-1,j+1]
            elif 67.5 <= angle < 112.5:
                q, r = grad_magnitude[i+1,j], grad_magnitude[i-1,j]
            elif 112.5 <= angle < 157.5:
                q, r = grad_magnitude[i-1,j-1], grad_magnitude[i+1,j+1]

            p = grad_magnitude[i][j]

            if p >= q and p >= r:
                points[i][j] = p

    points[points < h1] = 0 # non edge
    points[(points >= h1) & (points <= h2)] = 1 # weak edge
    points[points > h2] = 255 # strong edge

    stack = list(zip(*np.where(points == 255)))
    i = 0

    while stack:
        i, j = stack.pop()

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:

                ni = i + di
                nj = j + dj

                if 0 <= ni < h and 0 <= nj < w:

                    if points[ni, nj] == 1:
                        points[ni,  nj] = 255
                        stack.append((ni, nj))

    points[points == 1] = 0
    
    # cv.imshow(f'canny', points)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return points


def Hough_Transform(canny, dia_len):
    edge_points = np.nonzero(canny > 0)
    y, x = edge_points[0], edge_points[1]

    accumulator = np.zeros((180,dia_len*2))

    min_dist = -dia_len
    angle = -90

    for i in range(180):

        p_values =  np.round(x*np.cos([np.deg2rad(angle+i)]) + y*np.sin([np.deg2rad(angle+i)]),decimals=0)
        p_values = p_values.astype(int)
    
        for col in range(dia_len*2):

            dist_value = min_dist + col

            count = np.size(p_values[p_values == dist_value])
            
            if count > 0:
                accumulator[i][col] = count
            
    global_max = np.max(accumulator).astype(int)
    threshold = int(global_max*0.7)

    maxima = np.where(accumulator[:,:] >= threshold)
    m_a, m_b = maxima[0], maxima[1]


    indices = np.arange(len(m_a))

    i = 0

    while i < indices.shape[0]:

        temp_indices = indices[indices>i] 
        idx = np.where((m_a[temp_indices] == m_a[i]) | (np.abs(m_a[temp_indices] - m_a[i]) < 5))[0]
        idx = idx + (i+1)

        idx = idx[np.abs(m_b[idx] - m_b[i]) < 5]
        indices = np.delete(indices, (idx-1))
        i += 1

    
    maxima = (m_a[indices], m_b[indices])

    theta, r = angle + maxima[0], min_dist + maxima[1]

    return theta, r


def compute_angle(theta):

    angle = np.diff(theta)
    angle = (angle + 180) % 360 - 180
    angle = np.abs(angle)

    return angle

def draw_lines(theta, r, image, f_name):
    
    # polar to cartesian coordinates
    a , b = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))

    x0, y0 = r * a, r * b
    x1 = (x0 + 1000*(-b)).astype(int)
    y1 = (y0 + 1000*(a)).astype(int)
    x2 = (x0 - 1000*(-b)).astype(int)
    y2 = (y0 - 1000*(a)).astype(int)

    for l in range(len(theta)):
        cv.line(image,(x1[l],y1[l]),(x2[l],y2[l]),(255,0,0),1)

    
    cv.imshow(f'{f_name}',image)
    cv.waitKey(0)
    cv.destroyAllWindows()




def test_task_c1(folder_name):
    # assume that this folder name has a file list.txt that contains the annotation.
    task1_data = pd.read_csv(folder_name + "list.txt")
    # task1_data = pd.read_csv(folder_name + "/list.txt")
    # Write code to read in each image
    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image
    
    for f in task1_data["FileName"]:
        
        image = cv.imread(f"{folder_name}/" + f, cv.IMREAD_GRAYSCALE)

        if image is None:
            print("Error: Image not found or unable to read.")
            return 

        # Write code to process the image
        height, width = image.shape
        dia_len = int(np.sqrt(height**2 + width**2))

        # points = Canny(image, dia_len)
        points = cv.Canny(image, 50, 100)
        theta, r = Hough_Transform(points, dia_len)
        angle = compute_angle(theta)

        # draw_lines(theta, r, image, f)

        print(angle)

      
    total_error = None
    return total_error

# compute kernel
def epanechnikov_kernel(w, h):
    center_x, center_y = w//2, h//2
    y_indices, x_indices = np.indices((h, w))
    max_radius = np.sqrt((w / 2)**2 + (h / 2)**2)
    grid = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)/max_radius
    weights = np.where(grid <= 1, 3/4*(1-grid**2), 0)
    return weights

 # compute intensity histogram
def calculate_histogram(region, kernel):
    hist = np.bincount(region.flatten(), weights=kernel.flatten(), minlength=256)
    norm_hist = cv.normalize(hist, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    return np.float32(norm_hist)
    
def compare_histograms(hist1, hist2):
    return cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)


def template_matching(image, temps, temps_hist, kernels):
    
    step = 5
    search_size = 40
    boxes = []
    scores = []

    for t in range(temps.shape[0]):
        best_score = -1
        best_match = None
        best_temp = None

        for i in range(temps.shape[1]):
            template = temps[t][i]
        
            result_ncc = cv.matchTemplate(image, template, cv.TM_CCORR_NORMED)
            _, _, _, max_loc = cv.minMaxLoc(result_ncc)

            x, y = max_loc
            h, w = template.shape[:2]

            search_x_start = max(0, x - search_size // 2)
            search_y_start = max(0, y - search_size // 2)
            search_x_end = min(image.shape[1] - w, x + search_size // 2)
            search_y_end = min(image.shape[0] - h, y + search_size // 2)

            for check_y in range(search_y_start, search_y_end, step):
                for check_x in range(search_x_start, search_x_end, step):
                    if check_y + h > image.shape[0] or check_x + w > image.shape[1]:
                        continue

                    region = image[check_y:check_y+h, check_x:check_x+w]
                    region_hist = calculate_histogram(region, kernels[t][i])
                    hist_score = compare_histograms(temps_hist[t][i], region_hist)
                    ncc_score = result_ncc[check_y, check_x]

                    combined_score = 1 * ncc_score + 0 * hist_score

                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = (check_x, check_y)
                        best_temp = template
                    
        h, w = best_temp.shape[:2]
        scores.append(best_score)
        boxes.append([best_match[0], best_match[1], best_match[0]+w, best_match[1]+h])

    return boxes, scores

def NMS(boxes, treshold=0):
        indices = np.arange(len(boxes))

        areas = (boxes[indices, 2] - boxes[indices, 0]+1)*(boxes[indices, 3] - boxes[indices, 1]+1)

        i = 0

        while i < indices.shape[0]-1:
            temp_indices = indices[indices>i]
            
            xx1 = np.maximum(boxes[i][0], boxes[temp_indices, 0])
            yy1 = np.maximum(boxes[i][1], boxes[temp_indices, 1])
            xx2 = np.minimum(boxes[i][2], boxes[temp_indices, 2])
            yy2 = np.minimum(boxes[i][3], boxes[temp_indices, 3])

            w = np.maximum(0, xx2-xx1+1)
            h = np.maximum(0, yy2-yy1+1)
           
            overlap = (w*h)/ areas[temp_indices]


            indices = indices[~np.isin(indices, temp_indices[overlap > treshold])] 

            i += 1    

        return indices


def test_task_c2(icon_dir, test_dir):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    templates = []
    template_sizes=[0.125, 0.25, 0.375, 0.5, 1]

    folder = os.listdir(icon_dir)
    for file in folder:
        templates.append(cv.imread(icon_dir + file, cv.IMREAD_GRAYSCALE))

    for file in os.listdir(test_dir):
        test_image= cv.imread(f"{test_dir}{file}")

        if test_image is None:
            print("Error: Image not found or unable to read.")
            return 
        
        gray = cv.cvtColor(test_image, cv.COLOR_RGB2GRAY)

        multi_templates = np.zeros([len(templates),len(template_sizes)], dtype=object)
        templates_kernel = np.zeros([len(templates),len(template_sizes)], dtype=object)
        templates_hist = np.zeros([len(templates),len(template_sizes)], dtype=object)

        for t in range(len(templates)):
            template = templates[t]

            for s in range(len(template_sizes)):
                
                # compute kernel and template intensity histogram
                temp = cv.resize(template , None, fx = template_sizes[s], fy = template_sizes[s], interpolation = cv.INTER_CUBIC)
                kernel = epanechnikov_kernel(temp.shape[1], temp.shape[0])
                temp_hist = calculate_histogram(temp, kernel)

                multi_templates[t][s] = temp
                templates_kernel[t][s] = kernel
                templates_hist[t][s] = temp_hist


        # template matching
        boxes, scores = template_matching(gray, multi_templates, templates_hist, templates_kernel)


        arr = np.column_stack((np.array(boxes), np.arange(len(boxes)), np.array(scores)))
        
    
        set = arr[arr[:,5].argsort()[::-1]]

        indices = NMS(set[:, :4])

        boxes = set[indices].astype(np.int32).tolist()

        print(f"{file}:")

        for b in boxes:
            cv.rectangle(test_image, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)

            cv.putText(
                test_image,
                folder[b[4]],
                (b[0], b[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

            print(f"{folder[b[4]]}: {b[0]} {b[1]} {b[2]} {b[3]}")
            
        cv.imshow(f'{file}', test_image)
        cv.waitKey(0)
        cv.destroyAllWindows()


    
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
