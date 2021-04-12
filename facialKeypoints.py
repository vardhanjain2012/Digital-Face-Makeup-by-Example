# import the necessary packages
from imutils import face_utils
import dlib
import cv2
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
def keypoints(orignalImage):
	p = "facialRecognition/shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(p)
	image = orignalImage.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	points = []
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (x, y) in shape:
			points.append((x, y))
			cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
	assert len(points)==68
	size = (image.shape[0],image.shape[1])
	cv2.imshow("Output", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return points

if __name__ == '__main__':
    subject = cv2.imread('sampleImages/s2.jpg', cv2.IMREAD_COLOR)
    keypoints(subject)
