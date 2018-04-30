# import the necessary packages
# pip install --upgrade imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import glob,os
import random, sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")

ap.add_argument("-d", "--image_dir", required=True,
	help="path to input parent directory containing sub_dir of images")

ap.add_argument("-w", "--face-width", default=160, required=False,help="face cropping size")

ap.add_argument("-o", "--output_dir", required=True,
	help="path to output directory of images")

args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner

# detector object says features in faces has to be looked for.
detector = dlib.get_frontal_face_detector()
# To detect face pretrained weights of neural networks is given by dlib team.
predictor = dlib.shape_predictor(args["shape_predictor"])

# Face aligner does align faces so that all face images in dataset has eyes ,nose and lips in same coordinates.
fa = FaceAligner(predictor, desiredFaceWidth=int(args["face_width"]))


# users_images is list containing paths of images
users_images = []
#search for only images which are in png and jpg format
types = ['*.png','*.PNG', '*.JPG','*.jpg']

#listing all the subdirectories inside the parent directory given as argument
subdir = glob.glob(args["image_dir"]+"/*/")

# gathering paths of input images
for dirs in subdir:
	for file_types in types:
		files = glob.glob(dirs+'/'+file_types)
		users_images.extend(files)

#Create the new output directory where all cropped faces to be stored 

if not os.path.isdir(args["output_dir"]):
	os.makedirs(args["output_dir"])

for idx, i in enumerate(users_images):
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(i)
	if image is not None:
		# Read image and resize
		image = imutils.resize(image, width=800)
		# convert image into gray scale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# Rectangular box around face
		rects = detector(gray, 2)
	
		# loop over the face detections
		for rect in rects:
		
			(x, y, w, h) = rect_to_bb(rect)
			# Align the faces after cropping			
			#faceOrig = imutils.resize(image[y:y + h, x:x + w], width=160)
			faceAligned = fa.align(image, gray, rect)
			# saved the cropped file
			cv2.imwrite(os.path.join(args["output_dir"] ,( str(idx) + "_"+str(random.randint(1,1000000)) + ".jpg")), faceAligned)
	else :
		print("Courrupted image path :" + i)