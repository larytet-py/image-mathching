#!/usr/bin/python 

'''color-palette.py
Returs color palette of the image
Usage:
  color-palette.py -h | --help
  color-palette.py -f <FILENAME>
  color-palette.py -f <FILENAME> -c <FILENAME>
Example:
    color-palette.py -f favicon.bmp
    color-palette.py -f favicon.bmp -c favicon1.bmp
   
Options:
  -h --help               Show this screen.
  -f --file=<FILENAME>    Image to process
  -c --compare=<FILENAME> Image to compare
'''

# Based on https://stackoverflow.com/questions/18801218/build-a-color-palette-from-image-url

import numpy
from PIL import Image
from docopt import docopt
import logging
import struct

def palette(img):
  """
  Return palette in descending order of frequency
  """
  '''
  arr = numpy.asarray(img)
  palette, index = numpy.unique(asvoid(arr).ravel(), return_inverse=True)
  palette = palette.view(arr.dtype).reshape(-1, arr.shape[-1])
  count = numpy.bincount(index)
  order = numpy.argsort(count)
  return palette[order[::-1]]
  '''
  points = {}
  arr = numpy.asarray(img)
  data = asvoid(arr).ravel()
  for point in data:

    while len(point) < 4:
      point = b'\x00' + point
    point_color = struct.unpack(">L", point)[0]

    if point_color in points.keys():
      points[point_color] += 1
    else:
      points[point_color] = 1

  return len(data), points

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
  return arr.view(numpy.dtype((numpy.void, arr.dtype.itemsize * arr.shape[-1])))

def normalize_color_palette(image_size, color_palette):
  for color in color_palette.keys():
    color_palette[color] = (1.0*color_palette[color])/image_size

def print_color_palette(color_palette):
  for color in sorted(color_palette.keys()):
     print("{0:6X} {1:1.4f}".format(color, color_palette[color]))

def palette_distance(color_palette1, color_palette2):
  color_palette_sorted1 = sorted(color_palette1.keys())
  color_palette_sorted2 = sorted(color_palette2.keys())
  palette_size = min(len(color_palette1), len(color_palette2))
  distance = 0
  for idx in range(palette_size):
    c1 = color_palette_sorted1[idx]
    c2 = color_palette_sorted2[idx]
    distance += (c1-c2)**2
    distance += (color_palette1[c1]-color_palette2[c2])**2
  return distance

if __name__ == '__main__':
  arguments = docopt(__doc__, version='0.1')
  logging.basicConfig()    
  logger = logging.getLogger('hamming')
  logger.setLevel(logging.INFO)  
  image_file = arguments['--file']

  image = Image.open(image_file, 'r').convert('RGB')
  image_size, color_palette = palette(image)
  normalize_color_palette(image_size, color_palette)

  compare_file = arguments['--compare']
  if compare_file is None:
    print_color_palette(color_palette)
    exit(0)

  image = Image.open(compare_file, 'r').convert('RGB')
  image_size, color_palette_compare = palette(image)
  normalize_color_palette(image_size, color_palette_compare)

  distance = palette_distance(color_palette, color_palette_compare)
  if distance == 0:
    print("Perfect macth")
  else:
    print("Distance {0}".format(distance))


