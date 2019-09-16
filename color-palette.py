#!/usr/bin/python 

'''color-palette.py
Returs color palette of the image
Usage:
  color-palette.py -h | --help
  color-palette.py -f <FILENAME>
  color-palette.py -f <FILENAME> -c <FILENAME>
Example:
    color-palette.py -f favicon.bmp [--distance=<NUMBER>]
    color-palette.py -f favicon.bmp -c favicon1.bmp [--distance=<NUMBER>]
   
Options:
  -h --help               Show this screen.
  -f --file=<FILENAME>    Image to process
  -c --compare=<FILENAME> Image to compare
  -d --distance=<NUMBER>  Maximum RGB distance between matching colors
'''

import sys
import numpy
from PIL import Image
from docopt import docopt
from collections import namedtuple

import logging
import struct
import math

import colormath.color_objects 
import colormath.color_conversions 
import colormath.color_diff 


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

def matching_colors(c1, c2, max_distance):
  return (abs(c1[0]-c2[0]) < max_distance) and (abs(c1[1]-c2[1]) < max_distance) and (abs(c1[2]-c2[2]) < max_distance)

def find_matching_color(color, palette, max_distance):
  for c in palette.keys():
    if color != c and matching_colors(color, c, max_distance):
      return True, c
  return False, None

def palette(image, max_rgb_distance):
  palette = {}
  arr = numpy.asarray(image)
  data = asvoid(arr).ravel()
  for point in data:    
    point_color = struct.unpack("BBB", point)
    #point_color_rgb = colormath.color_objects.sRGBColor(point_color[0], point_color[1], point_color[2])
    if point_color in palette.keys():
      palette[point_color] += 1
    else:
      is_match, matching_color = find_matching_color(point_color, palette, max_rgb_distance)
      if is_match:
        palette[matching_color] += 1
      else:
        palette[point_color] = 1

  return len(data), palette

def rgb_distance(color1_rgb, color2_rgb):
  color1_lab = colormath.color_conversions.convert_color(color1_rgb, colormath.color_objects.LabColor);
  color2_lab = colormath.color_conversions.convert_color(color2_rgb, colormath.color_objects.LabColor);
  delta_e = colormath.color_diff.delta_e_cie2000(color1_lab, color2_lab);
  return delta_e

def normalize_color_palette(image_size, color_palette):
  '''
  Color = namedtuple('Color', ['count', 'distance', 'match'])
  for color1 in color_palette.keys():
    color_distance_min = sys.maxsize
    color_match = None
    for color2 in color_palette.keys():
      if color1 == color2:
        continue
      color_distance = rgb_distance(color1, color2)
      if color_distance_min < color_distance:
        color_distance_min, color_match = color_distance, color2

    color_palette[color1] = Color(color_palette[color1], color_distance_min, color_match)
  '''

  for color in color_palette.keys():
    color_palette[color] = (1.0*color_palette[color])/image_size

def print_color_palette(color_palette):
  for color in sorted(color_palette.keys()):
     print("{0} {1:1.4f}".format(color, color_palette[color]))

def palette_distance(color_palette1, color_palette2):
  color_palette_sorted1 = sorted(color_palette1.keys())
  color_palette_sorted2 = sorted(color_palette2.keys())
  palette_size = min(len(color_palette1), len(color_palette2))
  distance = 0
  for idx in range(palette_size):
    c1 = color_palette_sorted1[idx]
    c2 = color_palette_sorted2[idx]
    distance += (c1[0]-c2[0])**2
    distance += (c1[1]-c2[1])**2
    distance += (c1[2]-c2[2])**2
    distance += (color_palette1[c1]-color_palette2[c2])**2
  distance = math.sqrt(distance/palette_size)
  return distance

if __name__ == '__main__':
  arguments = docopt(__doc__, version='0.1')
  logging.basicConfig()    
  logger = logging.getLogger('hamming')
  logger.setLevel(logging.INFO)  
  image_file = arguments['--file']

  rgb_distance_str = arguments['--distance']
  if rgb_distance_str is None:
    rgb_distance_str = "20"
  rgb_distance =  int(rgb_distance_str, 10)

  image = Image.open(image_file, 'r').convert('RGB')
  image_size, color_palette = palette(image, rgb_distance)
  normalize_color_palette(image_size, color_palette)

  compare_file = arguments['--compare']
  if compare_file is None:
    print(image_file)
    print_color_palette(color_palette)
    exit(0)

  image = Image.open(compare_file, 'r').convert('RGB')
  image_size, color_palette_compare = palette(image, rgb_distance)
  normalize_color_palette(image_size, color_palette_compare)

  distance = palette_distance(color_palette, color_palette_compare)
  if distance == 0:
    print("{0}, {1} perfect match".format(image_file, compare_file))
  else:
    print("{0}, {1} Distance {2:08.0f}".format(image_file, compare_file, distance))


