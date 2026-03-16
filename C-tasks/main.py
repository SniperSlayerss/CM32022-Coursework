import argparse
import os
import time
import numpy as np
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



        
def Canny(image, h1=80, h2=150):        

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
                q, r = grad_magnitude[i-1,j-1], grad_magnitude[i+1,j+1]
            elif 67.5 <= angle < 112.5:
                q, r = grad_magnitude[i+1,j], grad_magnitude[i-1,j]
            elif 112.5 <= angle < 157.5:
                q, r = grad_magnitude[i+1,j-1], grad_magnitude[i-1,j+1]

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

    # canny = cv.Canny(image, 50, 100)
    edge_points = np.nonzero(canny > 0)
    y, x = edge_points[0], edge_points[1]

    accumulator = np.zeros((180,dia_len*2))

    min_dist = -dia_len
    angle = -90
    # angle = 0

    for i in range(180):

        p_values =  np.round(x*np.cos([np.deg2rad(angle+i)]) + y*np.sin([np.deg2rad(angle+i)]),decimals=0)
        p_values = p_values.astype(int)
    
        for col in range(dia_len*2):

            dist_value = min_dist + col

            count = np.size(p_values[p_values == dist_value])
            
            if count > 0:
                accumulator[i][col] = count

    j = 0

    maxima = []
    temp_accumulator = accumulator

    while j < 2:
        
        global_max = np.max(temp_accumulator)
        # params = np.where(accumulator == global_max)

        params = np.argwhere(temp_accumulator == global_max)

        for p in params:
            if p[0] not in map(lambda x: x[0] ,maxima):
                maxima.append(p)
                j += 1

            if j >= 2:
                break

        temp_accumulator[temp_accumulator == global_max] = 0
        
    maxima = np.array((maxima)).reshape(-1, 2).transpose()


    theta, r = angle + maxima[0], min_dist + maxima[1]

    return theta, r
  

def compute_angle(theta):
    angle = np.diff(theta)

    angle = (angle + 180) % 360 - 180

    angle = np.abs(angle)

    if np.all(theta < 0):
        angle = 180 - angle

    return angle[0]


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

# compute kernel
def epanechnikov_kernel(w, h):
    center_x, center_y = w//2, h//2
    y_indices, x_indices = np.indices((h, w))
    max_radius = np.sqrt((w / 2)**2 + (h / 2)**2)
    grid = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)/max_radius
    weights = np.where(grid <= 1, 3/4*(1-grid**2), 0)
    return weights.flatten()

 # compute intensity histogram
def calculate_histogram(region, kernel):
    hist = np.bincount(region.ravel(), weights=kernel, minlength=256)
    norm_hist = cv.normalize(hist, None, alpha=0, beta=255, norm_type=cv.NORM_L1)
    return np.float32(norm_hist)

    
def compare_histograms(hist1, hist2):
    return cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)


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

                    region = image[check_y:check_y+h, check_x:check_x+w]

                    
                    region_hist = calculate_histogram(region, kernels[t][i])
                    hist_score = compare_histograms(temps_hist[t][i], region_hist)
                    ncc_score = result_ncc[check_y, check_x]

                    combined_score = alpha * ncc_score + (1-alpha) * hist_score

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

    cv.imshow(f'{file}', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


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
            box2_left, box2_top, box2_right, box2_bottom =  left[j], top[j], right[j], bottom[j]

            xx1 = np.maximum(box1_left, box2_left)
            yy1 = np.maximum(box1_top, box2_top)
            xx2 = np.minimum(box1_right, box2_right)
            yy2 = np.minimum(box1_bottom, box2_bottom)

            w = np.maximum(0, xx2-xx1+1)
            h = np.maximum(0, yy2-yy1+1)

            box1_area = (box1_right - box1_left) * (box1_bottom - box1_top)
            box2_area = (box2_right - box2_left) * (box2_bottom - box2_top)

            intersection = w*h
            union = box1_area + box2_area - intersection

            if union > 0:
                iou = intersection/ union
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

        template_scales=[0.125, 0.25, 0.375, 0.5, 1]
       

        for file in os.listdir(image_dir):
            test_image= cv.imread(f"{image_dir}{file}")


            if test_image is None:
                print("Error: Image not found or unable to read.")
                return 
            
            gray = cv.cvtColor(test_image, cv.COLOR_RGB2GRAY)

            multi_templates = np.zeros([len(templates),len(template_scales)], dtype=object)
            templates_kernel = np.zeros([len(templates),len(template_scales)], dtype=object)
            templates_hist = np.zeros([len(templates),len(template_scales)], dtype=object)

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
            set = arr[arr[:,5].argsort()[::-1]]

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

        annotate_f =  pd.read_csv(f"{annotate_dir}{f}")
        classidx = annotate_f["classname"].tolist()
        top = annotate_f["top"].tolist()
        left = annotate_f["left"].tolist()
        bottom = annotate_f["bottom"].tolist()
        right = annotate_f["right"].tolist()

        boxes = pred[f[:-4]]

        tp, fp, fn = get_metrics(
            icon_folder,
            boxes,
            classidx,
            top,
            left,
            bottom,
            right,
            iou_threshold=0.85
            )
        
        t_tp += tp
        t_fp += fp
        t_fn += fn

    acc = t_tp / (t_tp + t_fp + t_fn + 1e-6)   
    tpr = t_tp / (t_tp + t_fn + 1e-6)
    fpr = t_fp / (t_fp+ t_tp + 1e-6)
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
