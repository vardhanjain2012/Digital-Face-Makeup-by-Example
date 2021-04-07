import cv2
import numpy as np

img = cv2.imread("./sampleImages/s1.jpg", cv2.IMREAD_COLOR)

cv2.imshow("Cute Kitens color", img)

img = cv2.imread("./sampleImages/s1.jpg", cv2.IMREAD_GRAYSCALE)

# cv2.imshow("Cute Kitens", img)

low_sigma = cv2.GaussianBlur(img,(5,5), 1)

# cv2.imshow("low_sigma", low_sigma)

high_sigma = cv2.GaussianBlur(img,(5,5), 1.6)

# cv2.imshow("high_sigma", high_sigma)

dog = low_sigma - high_sigma

cv2.imshow("dog", dog)

tau = 0.98

xdog = low_sigma - tau*high_sigma

cv2.imshow("xdog", xdog)

maxI = 255

def threshold(src, threshold, ):
	return np.uint8((src>=(maxI*threshold))*maxI)

def softThreshold(src, threshold, psi):
	return np.uint8(np.add((src>=(maxI*threshold))*maxI, (src<(maxI*threshold))*(np.add(np.ones(src.shape), np.tanh(psi*((src/maxI) - threshold))))*maxI))

def blurred(src):
	return

def scaledDog(src):
	return

def sharpenedDog(src):
	return


thresholded = threshold(dog, 0.99)

cv2.imshow("thresholded", thresholded)

softThresholded = softThreshold(xdog, 0.5, 2.0)

cv2.imshow("softThresholded", softThresholded)

cv2.waitKey(0)

cv2.destroyAllWindows()