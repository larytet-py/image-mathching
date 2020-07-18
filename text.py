import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

class TextBox():
    def __init__(self, confidence, startX, startY, endX, endY):
        values = {}
        values['confidence'] = confidence
        values['startX'] = startX
        values['startY'] = startY
        values['endX'] = endX
        values['endY'] = endY
        self.values = values

    

class Image():
    def __init(self, image, model, confidence, logger):
        '''
        Image can be a result of cv2.imread or a filename
        Use 0.9 for confidence
        Use model from https://github.com/larytet-py/image-mathching/releases/download/base-line/frozen_east_text_detection.pb
        '''
        if isinstance(image, basestring):
            logger.info(f"Loading image from a file {image}")
            image = cv2.imread(image)
        if isinstance(model, basestring):
            logger.info(f"Loading EAST text detector from a file {model}")
            model = cv2.dnn.readNet(model)

        self.image, self.model, self.logger = image, model, logger

    def get_text_areas(self):
        '''
        return list of areas containing text 
        This code is straight from https://developpaper.com/opencv-text-detection/
        '''
        image = self.image
        (H, W) = image.shape[:2]
        (newW, newH) = (alignment32(H), alignment32(W))

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
        self.logger.info("Loading EAST text detector...")
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
        self.model.setInput(blob)
        (scores, geometry) = self.model.forward(layerNames)
        end = time.time()

        # show timing information on text prediction
        logger.info("Text detection took {:.6f} seconds".format(end - start))

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]

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

        return text_boxes

    def collage(text_boxes):
        image = self.image

        confidences = []
        rectangles = []
        for (confidence, startX, startY, endX, endY) in text_boxes:
            confidences.append(confidence)
            rectangles.append((startX, startY, endX, endY))
            # see https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
        rectangles = non_max_suppression(np.array(rectangles), probs=confidences)
        logger.info(f"Discovered {len(text boxes)},  after suppression remained {len(rectangles)} text boxes")

        w, h = 100, 60
        collage = 255*np.ones(shape=[w, h, 3], dtype=np.uint8)
        for (startX, startY, endX, endY) in rectangles:
            rectangle = image[startY:endY, startX:endX] #cv2.cv.GetSubRect(image, (startX, startY, endX, endY))
            rectangle = cv2.resize(rectangle.copy(), dsize=(h, w), interpolation=cv2.INTER_CUBIC)
            collage = np.vstack([collage, rectangle])
        
        return collage


def alignment32(x): 
	'''
	EAST requires aligned image dimensions 
	'''
	return (x+32) & ~31
