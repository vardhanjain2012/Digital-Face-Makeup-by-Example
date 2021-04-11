import cv2
import numpy as np
img = cv2.imread("./sampleImages/s2.jpg", cv2.IMREAD_COLOR)

cv2.imshow("colored", img)

img = cv2.imread("./sampleImages/s2.jpg", cv2.IMREAD_GRAYSCALE)

# cv2.imshow("Gray", img)


sigma = 0.8
k = 1.6

low_sigma = cv2.GaussianBlur(img, (7, 7), sigma)

# cv2.imshow("low_sigma", low_sigma)

high_sigma = cv2.GaussianBlur(img, (7, 7), sigma*k)

# cv2.imshow("high_sigma", high_sigma)

dog = low_sigma - high_sigma

# cv2.imshow("dog", dog)

tau = 0.98

xdog1 = low_sigma - (tau*high_sigma)

rho = 18

xdog2 = low_sigma + rho*dog 

# cv2.imshow("xdog", xdog)

maxI = 255

def threshold(src, threshold):
	return np.uint8((src>=(maxI*threshold))*maxI)

def softThreshold(src, threshold, phi):
	return np.uint8(np.add((src>=(maxI*threshold))*maxI, (src<(maxI*threshold))*(np.add(np.ones(src.shape), np.tanh(phi*((src/maxI) - threshold))))*maxI))


epsilon = 0.50
phi = 1.0
thresholded = threshold(dog, epsilon)

cv2.imshow("thresholded", thresholded)

softThresholded = softThreshold(xdog2, epsilon, phi)

cv2.imshow("softThresholded", softThresholded)

bilateral = cv2.bilateralFilter(img, 15, 75, 75)

cv2.imshow('bilateral.04', bilateral)

cv2.waitKey(0)

cv2.destroyAllWindows()