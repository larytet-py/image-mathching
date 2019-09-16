#!/usr/bin/python 

'''color-palette.py
Returs color palette of the image
Usage:
  color-palette.py -h | --help
  color-palette.py -f <FILENAME>
  color-palette.py -f <FILENAME> -c <FILENAME>
Example:
    color-palette.py -f favicon.bmp [--size=<NUMBER>]
    color-palette.py -f favicon.bmp -c favicon1.bmp [--size=<NUMBER>]
   
Options:
  -h --help               Show this screen.
  -f --file=<FILENAME>    Image to process
  -c --compare=<FILENAME> Image to compare
  -s --size=<NUMBER>      Number of colors in palette
'''

import numpy
from PIL import Image
from docopt import docopt
import logging
import struct
import math


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
    point_color = (point[0], point[1], point[2])
    if point_color in points.keys():
      points[point_color] += 1
    else:
      points[point_color] = 1

  return len(data), points

import colormath.color_objects 
import colormath.color_conversions 
import colormath.color_diff 

def rgb_distance(color1_rgb, color2_rgb):
  color1_lab = colormath.color_conversions.convert_color(color1_rgb, LabColor);
  color2_lab = colormath.color_conversions.convert_color(color2_rgb, LabColor);
  delta_e = colormath.color_diff.delta_e_cie2000(color1_lab, color2_lab);
  return delta_e

print "The difference between the 2 color = ", delta_e

def normalize_color_palette(image_size, color_palettem, palette_size):

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
  distance = math.sqrt(distance/palette_size)
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
    print(image_file)
    print_color_palette(color_palette)
    exit(0)

  image = Image.open(compare_file, 'r').convert('RGB')
  image_size, color_palette_compare = palette(image)
  normalize_color_palette(image_size, color_palette_compare)

  distance = palette_distance(color_palette, color_palette_compare)
  if distance == 0:
    print("{0}, {1} perfect match".format(image_file, compare_file))
  else:
    print("{0}, {1} Distance {2:08.0f}".format(image_file, compare_file, distance))


