# import the necessary packages
# pip install --upgrade imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import glob,os
import random




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-d", "--image_dir", required=True,
	help="path to input directory containing all images")
ap.add_argument("-w", "--face-width", required=True,help="face cropping size")

ap.add_argument("-o", "--output_dir", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=int(args["face_width"]))

users_images = []
types = ('*.png','*.PNG', '*.JPG','*.jpg')
for files in types:
	users_images.extend(glob.glob(os.path.join(args["image_dir"],files)))
	

path = args["output_dir"]
if not (os.path.isdir(path)):
	os.makedirs(path)

for idx, i in enumerate(users_images):
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(i)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	rects = detector(gray, 2)
	
	# loop over the face detections
	for rect in rects:
		
		(x, y, w, h) = rect_to_bb(rect)
		
		#faceOrig = imutils.resize(image[y:y + h, x:x + w], width=160)
		faceAligned = fa.align(image, gray, rect)
		
		cv2.imwrite(os.path.join(path ,( str(idx) + "_"+str(random.randint(1,99999)) + ".jpg")), faceAligned)