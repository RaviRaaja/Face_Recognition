# import the necessary packages
from imutils import paths
import argparse
import cv2
import os
import random
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_dir", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=20.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

path = "/home/stan/face_landmark_detection/no_blur_"
if not (os.path.isdir(path)):
	os.makedirs(path)


# loop over the input images
for imagePath in paths.list_images(args["image_dir"]):

	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	

	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if fm >= args["threshold"]:
		#cv2.imwrite(os.path.join(path ,"face" + ".jpg"), image)
		cv2.imwrite(os.path.join(path ,( "face_"+str(random.randint(1,99999)) + ".jpg")), image)
		#text = "Blurry"

	