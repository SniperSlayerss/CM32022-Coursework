import cv2 as cv
import numpy as np


def compare_histograms(hist1, hist2):
    return cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)


def Hough_Transform(canny, dia_len):
    # canny = cv.Canny(image, 50, 100)
    edge_points = np.nonzero(canny > 0)
    y, x = edge_points[0], edge_points[1]

    accumulator = np.zeros((180, dia_len * 2))

    min_dist = -dia_len
    angle = -90
    # angle = 0

    for i in range(180):
        p_values = np.round(
            x * np.cos([np.deg2rad(angle + i)]) + y * np.sin([np.deg2rad(angle + i)]),
            decimals=0,
        )
        p_values = p_values.astype(int)

        for col in range(dia_len * 2):
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
            if p[0] not in map(lambda x: x[0], maxima):
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


def Canny(image, h1=80, h2=150):
    gaussian = cv.GaussianBlur(image, (5, 5), 0)

    # Sobel Operator - Image gradient
    sobelx = cv.Sobel(gaussian, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(gaussian, cv.CV_64F, 0, 1, ksize=5)

    # Gradient Magitude
    grad_magnitude = np.sqrt(sobelx**2 + sobely**2)
    grad_magnitude = cv.normalize(grad_magnitude, None, 0, 255, cv.NORM_MINMAX)

    # Gradient Orientation
    grad_orientation = np.rad2deg(np.arctan2(sobely, sobelx))
    grad_orientation[grad_orientation < 0] += 180

    h, w = image.shape
    points = np.zeros((h, w))

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = grad_orientation[i][j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = grad_magnitude[i, j + 1], grad_magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q, r = grad_magnitude[i - 1, j - 1], grad_magnitude[i + 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q, r = grad_magnitude[i + 1, j], grad_magnitude[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q, r = grad_magnitude[i + 1, j - 1], grad_magnitude[i - 1, j + 1]

            p = grad_magnitude[i][j]
            if p >= q and p >= r:
                points[i][j] = p

    points[points < h1] = 0  # non edge
    points[(points >= h1) & (points <= h2)] = 1  # weak edge
    points[points > h2] = 255  # strong edge

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
                        points[ni, nj] = 255
                        stack.append((ni, nj))

    points[points == 1] = 0

    # cv.imshow(f'canny', points)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return points
