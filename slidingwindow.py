# Purpose: Test if openCV packages are correctly installed. This should display an image.
# Source: https://opencv.org/get-started/
# Date: Sept 27, 2024


# Imports
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential


img = cv.imread('singlerow_template.JPG')
assert img is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
template = cv.imread('target.JPG', cv.IMREAD_GRAYSCALE)
assert template is not None, "template could not be read, check with os.path.exists()"
w, h = template.shape[::-1]

res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
threshold = 0.5
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv.imwrite('res50.png', img)
# cv.imshow('img', img)
# k = cv.waitKey(0)
