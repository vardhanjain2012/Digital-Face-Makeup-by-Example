from triangulation import generate_morphed_image
import cv2
import numpy as np
import random


imgPath1 = "./sampleImages/s1.jpg"
imgPath2 = "./sampleImages/s2.jpg"

# morphedImage = generate_morphed_image(imgPath1, imgPath2)
image = cv2.imread(imgPath1, cv2.IMREAD_COLOR)

cv2.imshow('image', image)

lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

L,A,B=cv2.split(lab_image)
cv2.imshow("L_Channel",L) # For L Channel
cv2.imshow("A_Channel",A) # For A Channel (Here's what You need)
cv2.imshow("B_Channel",B) # For B Channel
bilateral = cv2.bilateralFilter(L, 15, 75, 3)
cv2.imshow('bilateral.04', bilateral)

large_scale = L - bilateral

cv2.imshow('large_scale', large_scale)

cv2.waitKey(0)

cv2.destroyAllWindows()