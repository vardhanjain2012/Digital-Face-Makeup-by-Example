from facialKeypoints import keypoints
import cv2
import numpy as np
import random


# Check if a point is inside a rectangle
def rect_contains(rect, point):

	if point[0] < rect[0]:
		return False
	elif point[1] < rect[1]:
		return False
	elif point[0] > rect[2]:
		return False
	elif point[1] > rect[3]:
		return False
	return True

# Write the delaunay triangles into a file
def draw_delaunay(f_w, f_h, subdiv, dictionary1):

	list4 = []

	triangleList = subdiv.getTriangleList()
	r = (0, 0, f_w, f_h)

	for t in triangleList :
		pt1 = (int(t[0]), int(t[1]))
		pt2 = (int(t[2]), int(t[3]))
		pt3 = (int(t[4]), int(t[5]))

		if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
			list4.append((dictionary1[pt1],dictionary1[pt2],dictionary1[pt3]))

	dictionary1 = {}
	return list4

def make_delaunay(f_w, f_h, theList):

	# Make a rectangle.
	rect = (0, 0, f_w, f_h)

	# Create an instance of Subdiv2D.
	subdiv = cv2.Subdiv2D(rect)

	# Make a points list and a searchable dictionary. 
	theList = theList
	points = [(int(x[0]),int(x[1])) for x in theList]
	dictionary = {x[0]:x[1] for x in list(zip(points, range(76)))}
	
	# Insert points into subdiv
	for p in points :
		subdiv.insert(p)

	# Make a delaunay triangulation list.
	list4 = draw_delaunay(f_w, f_h, subdiv, dictionary)
   
	# Return the list.
	return list4

def drawLines(img, points, listd):
	img1 = img.copy()
	for (x, y, z) in listd:
		pt1 = (int(points[x][0]), int(points[x][1]))
		pt2 = (int(points[y][0]), int(points[y][1]))
		pt3 = (int(points[z][0]), int(points[z][1]))
		cv2.line(img1, pt3, pt1, (255, 255, 255), 1, 8, 0)
		cv2.line(img1, pt2, pt1, (255, 255, 255), 1, 8, 0)
		cv2.line(img1, pt3, pt2, (255, 255, 255), 1, 8, 0)

	cv2.imshow("Output", img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def apply_affine_transform(src, srcTri, dstTri, size) :
	
	# Given a pair of triangles, find the affine transform.
	warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
	
	# Apply the Affine Transform just found to the src image
	dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

	return dst


def morph_triangle(img1, img2, img, t1, t2, t, alpha) :
	# Find bounding rectangle for each triangle
	r1 = cv2.boundingRect(np.float32([t1]))
	r2 = cv2.boundingRect(np.float32([t2]))
	r = cv2.boundingRect(np.float32([t]))

	# Offset points by left top corner of the respective rectangles
	t1Rect = []
	t2Rect = []
	tRect = []

	for i in range(0, 3):
		tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
		t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
		t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

	# Get mask by filling triangle
	mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
	cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

	# Apply warpImage to small rectangular patches
	img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

	size = (r[2], r[3])
	warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
	warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

	# Alpha blend rectangular patches
	imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

	# Copy triangular region of the rectangular patch to the output image
	img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

def generate_morphed_image(img1, img2):

	points1 = keypoints(img1)
	# print(points, len(points))
	listd1 = make_delaunay(img1.shape[1], img1.shape[0], points1)
	# print(listd)
	drawLines(img1, points1, listd1)

	points2 = keypoints(img2)
	# print(points, len(points))
	listd2 = make_delaunay(img2.shape[1], img1.shape[0], points2)
	# print(listd)
	drawLines(img2, points2, listd2)
	morphed_image = np.zeros(img1.shape, dtype = img1.dtype)
	alpha = 1.0
	for i in range(len(listd1)):
		x = int(listd1[i][0])
		y = int(listd1[i][1])
		z = int(listd1[i][2])
		t1 = [points1[x], points1[y], points1[z]]
		t2 = [points2[x], points2[y], points2[z]]
		t = [points1[x], points1[y], points1[z]]
		morph_triangle(img1, img2, morphed_image, t1, t2, t, alpha)
	morphed_image_with_lines = morphed_image.copy()
	for i in range(len(listd1)):
		t = [points1[x], points1[y], points1[z]]
		pt1 = (int(t[0][0]), int(t[0][1]))
		pt2 = (int(t[1][0]), int(t[1][1]))
		pt3 = (int(t[2][0]), int(t[2][1]))
		cv2.line(morphed_image_with_lines, pt1, pt2, (255, 255, 255), 1, 8, 0)
		cv2.line(morphed_image_with_lines, pt2, pt3, (255, 255, 255), 1, 8, 0)
		cv2.line(morphed_image_with_lines, pt3, pt1, (255, 255, 255), 1, 8, 0)

	cv2.imshow("Morphed Image", morphed_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return morphed_image 


if __name__ == '__main__':
    subject = cv2.imread('sampleImages/s2.jpg', 1)
    target = cv2.imread('sampleImages/m1.jpg', 1)
    morphed_image = generate_morphed_image(subject, target)