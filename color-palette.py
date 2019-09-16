#!/usr/bin/python 

'''color-palette.py
Returs color palette of the image
Usage:
  color-palette.py -h | --help
  color-palette.py -f <FILENAME>
Example:
    color-palette.py -f favicon.bmp
   
Options:
  -h --help                 Show this screen.
  -f --file=<FILENAME>    Data set
'''

# Based on https://stackoverflow.com/questions/18801218/build-a-color-palette-from-image-url

import numpy
import Image
from docopt import docopt
import logging

def palette(img):
  """
  Return palette in descending order of frequency
  """
  arr = numpy.asarray(img)
  palette, index = numpy.unique(asvoid(arr).ravel(), return_inverse=True)
  palette = palette.view(arr.dtype).reshape(-1, arr.shape[-1])
  count = numpy.bincount(index)
  order = numpy.argsort(count)
  return palette[order[::-1]]

def asvoid(arr):
  """View the array as dtype np.void (bytes)
  This collapses ND-arrays to 1D-arrays, so you can perform 1D operations on them.
  http://stackoverflow.com/a/16216866/190597 (Jaime)
  http://stackoverflow.com/a/16840350/190597 (Jaime)
  Warning:
  >>> asvoid([-0.]) == asvoid([0.])
  array([False], dtype=bool)
  """
  arr = numpy.ascontiguousarray(arr)
  return arr.view(numpy.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


if __name__ == '__main__':
  arguments = docopt(__doc__, version='0.1')
  logging.basicConfig()    
  logger = logging.getLogger('hamming')
  logger.setLevel(logging.INFO)  
  data_file = arguments['--file']
  image = Image.open(data_file, 'r').convert('RGB')

  print(palette(img))
