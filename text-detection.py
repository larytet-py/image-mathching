# USAGE


'''text-detection.py
Collects areas containing text, returns 

Usage:
  text-detection.py -h | --help
  text-detection.py --image <FILENAME> --model <FILENAME> [--confidence <VALUE>] [--cache=<FILENAME>] [--collage=<FILENAME>] [--show]
   
Options:
  -h --help               Show this screen
  --image=<FILENAME>      Image to process
  --model=<FILENAME>      EAST model, for example https://github.com/larytet-py/image-mathching/releases/download/base-line/frozen_east_text_detection.pb 
                          [default: ./frozen_east_text_detection.pb]
  --confidence=<NUMBER>   Confidence level that an area contains text [default: 0.9]
  --cache=<FILENAME>      Cache filename to use [defualt: .text-detection.cache.yaml]
  --collage=<FILENAME>    Generate a collage of discovered text boxes
  --show                  Show the image with text boxes

Examples:
Generate "collages"
find ./images -name "*.png" | xargs -I FILE python3 text-detection.py --image FILE --model frozen_east_text_detection.pb  --confidence 0.6 --collage collages/FILE.collage.png

'''



import logging
from docopt import docopt
from collections import namedtuple

from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import os


def round32(x): 
	'''
	Dimensions must be multiple of 32 for EAST
	'''
	return (x+32) & ~31

def text_areas(image_filename, model_filename, confidence):
	'''
	return list of areas containing text 
	This code is straight from https://developpaper.com/opencv-text-detection/
	'''
	# load the input image and grab the image dimensions
	image = cv2.imread(image_filename)
	(H, W) = image.shape[:2]
	(newW, newH) = (round32(H), round32(W))

	ratioW = W / float(newW)
	ratioH = H / float(newH)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	logger.info("Loading EAST text detector...")
	net = cv2.dnn.readNet(model_filename)

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	# The magic (123.68, 116.78, 103.94) is RGB for image preprocessing 
	# before deep neural networks (dnn) kicks in
	# The numbers reflect "average pixel intensity across all images" in the 
	# data set used for the model
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)

	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()

	# show timing information on text prediction
	logger.info("Text detection took {:.6f} seconds".format(end - start))

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]

	TextBox = namedtuple('TextBox', ['confidence', 'startX', 'startY', 'endX', 'endY'])
	text_boxes = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < confidence:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = offsetX + (cos * xData1[x]) + (sin * xData2[x])
			endY = offsetY - (sin * xData1[x]) + (cos * xData2[x])
			startX = endX - w
			startY = endY - h

			# collect the bounding box coordinates and probability score 
			# fix the scale
			text_boxes.append(TextBox(scoresData[x], int(startX*ratioW), int(startY*ratioH), int(endX*ratioW), int(endY*ratioH)))

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	#boxes = non_max_suppression(np.array(rects), probs=confidences)
	return text_boxes

def generate_collage(image_filename, collage_filename):
	# load the input image 
	image = cv2.imread(image_filename)
	# loop over the bounding boxes

	confidences = []
	rectangles = []
	for (confidence, startX, startY, endX, endY) in text_boxes:
		confidences.append(confidence)
		rectangles.append((startX, startY, endX, endY))
		# see https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
	logger.info("Discovered {0} text boxes".format(len(rectangles)))
	rectangles = non_max_suppression(np.array(rectangles), probs=confidences)
	logger.info("After suppression remained {0} text boxes".format(len(rectangles)))

	w, h = 100, 60
	collage = 255*np.ones(shape=[w, h, 3], dtype=np.uint8)
	for (startX, startY, endX, endY) in rectangles:
		rectangle = image[startY:endY, startX:endX] #cv2.cv.GetSubRect(image, (startX, startY, endX, endY))
		rectangle = cv2.resize(rectangle.copy(), dsize=(h, w), interpolation=cv2.INTER_CUBIC)
		collage = np.vstack([collage, rectangle])
	
	logger.info("Writing to file {}".format(collage_filename))
	cv2.imwrite(collage_filename, collage)

	return collage


def show_text_boxes(image_filename):
	# load the input image 
	image = cv2.imread(image_filename)
	# loop over the bounding boxes
	for (_, startX, startY, endX, endY) in text_boxes:
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

	# show the output image
	cv2.imshow("Text Detection", image)

if __name__ == '__main__':
	arguments = docopt(__doc__, version='0.1')
	logging.basicConfig()    
	logger = logging.getLogger('text-detection')
	logger.setLevel(logging.DEBUG)
	image_filename = arguments['--image']

	cache_filename = arguments.get('--cache', ".text-detection.cache.yaml")
	collage_filename = arguments['--collage']
	model_filename = arguments.get('--model', "./frozen_east_text_detection.pb")
	confidence_str = arguments.get('--confidence', "0.6")
	confidence =  float(confidence_str)
	if collage_filename is not None and os.path.exists(collage_filename):
		logger.error("Skip existing file {}".format(collage_filename))
		exit(1)
	collage_dir = os.path.dirname(collage_filename)
	if not os.path.exists(collage_dir):
		logger.error("Output directory '{}' does not exist".format(collage_dir))
		exit(1)

	text_boxes = text_areas(image_filename, model_filename, confidence)
	if collage_filename is not None:
		collage = generate_collage(image_filename, collage_filename)
		if arguments["--show"]:
			cv2.imshow("Text Detection", collage)
			cv2.waitKey(0)

	if arguments["--show"]:
		show_text_boxes(image_filename)
		cv2.waitKey(0)

	exit(0)

