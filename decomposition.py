from triangulation import generate_morphed_image
import cv2
import numpy as np
import random

def decomposition(image):
	bilateral = cv2.bilateralFilter(image, 9, 75, 75)

	large_scale1 = image - bilateral

	# cv2.imshow('large_scale', large_scale1)


	# cv2.waitKey(0)

	# cv2.destroyAllWindows()

	return (image, large_scale1)



if __name__ == '__main__':
    subject = cv2.imread('sampleImages/s2.jpg', cv2.IMREAD_COLOR)
    decomposition(subject)