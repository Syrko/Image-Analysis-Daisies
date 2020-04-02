import cv2
import numpy as np
import slic_superpixels as ssp


def load_rgb_image(path):
    rgb_image = cv2.imread(path)
    return rgb_image


def rgb_to_lab(rgb):
    lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab_image


def rgb_to_gray(rgb):
    gray_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray_image


def lab_to_rgb(lab):
    rgb_image = cv2.cvtColor(lab, cv2.COLOR_LAB2LRGB)
    return rgb_image


def display_image(img, name="Image"):
    cv2.imshow(name, np.hstack([img]))
    cv2.waitKey(0)


def display_end_result(original_img, gray_img, colors):
    segments = ssp.calculate_slic_superpixels(gray_img, gray=True)
    (h, w) = gray_img.shape
    a = []
    b = []
    for i in range(h):
        rowA = []
        rowB = []
        for j in range(w):
            color_index = segments[i][j]
            rowA.append(colors[color_index][0])
            rowB.append(colors[color_index][1])
        a.append(rowA)
        b.append(rowB)

    a = np.array(a)
    b = np.array(b)
    predicted_image = np.stack((gray_img, a, b), axis=2)

    rgb = lab_to_rgb(predicted_image)

    cv2.imshow("Original - Colored LAB - Colored RGB", np.hstack([original_img, predicted_image, rgb]))
    cv2.imshow("Gray Image", gray_img)
    cv2.waitKey(0)
