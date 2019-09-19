# USAGE


'''text-detection.py
Collects areas containing text, returns 

Usage:
  text-detection.py -h | --help
  text-detection.py --image <FILENAME> --model <FILENAME> [--confidence <VALUE>] [--cache=<FILENAME>] [--collage=<FILENAME>]
Example:
  text-detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb 
   
Options:
  -h --help               Show this screen
  --image=<FILENAME>      Image to process
  --model=<FILENAME>      EAST model [default: ./frozen_east_text_detection.pb]
  --confidence=<NUMBER>   Confidence level that an area contains text [default 0.6]
  --cache=<FILENAME>      Cache filename to use [defualt: .text-detection.cache.yaml]
  --collage=<FILENAME>    Generate a collage of discovered text boxes
'''


import logging
from docopt import docopt
from collections import namedtuple

from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2


def round32(x): 
	return (x+32) & ~31

def text_areas(image_filename, model_filename, confidence):
	'''
	return list of areas containing text 
	This code is straight from https://developpaper.com/opencv-text-detection/
	'''
	# load the input image and grab the image dimensions
	image = cv2.imread(image_filename)
	(H, W) = image.shape[:2]
	# Must be multiple of 32 for EAST
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
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# collect the bounding box coordinates and probability score 
			# fix the scale
			text_boxes.append(TextBox(scoresData[x], startX*ratioW, startY*ratioH, endX*ratioW, endY*ratioH))

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	#boxes = non_max_suppression(np.array(rects), probs=confidences)
	return text_boxes

def collage(collage_filename):
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
	logger.setLevel(logging.INFO)  
	image_filename = arguments['--image']

	cache_filename = arguments.get('--cache', ".text-detection.cache.yaml")
	collage_filename = arguments['--collage']
	model_filename = arguments.get('--model', "./frozen_east_text_detection.pb")
	confidence_str = arguments.get('--confidence', "0.6")
	confidence =  float(confidence_str)

	text_boxes = text_areas(image_filename, model_filename, confidence)

	if collage_filename is None:
		exit(0)

	collage(collage_filename)
	cv2.waitKey(0)

