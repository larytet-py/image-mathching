#!/usr/bin/python 

'''color-palette.py
Returs color palette of the image
Usage:
  color-palette.py -h | --help
  color-palette.py -f <FILENAME>
  color-palette.py -f <FILENAME> -c <FILENAME>
Example:
    color-palette.py -f favicon.bmp [--distance=<NUMBER>] [--cache=<FILENAME>]
    color-palette.py -f favicon.bmp -c favicon1.bmp [--distance=<NUMBER>] [--cache=<FILENAME>]
   
Options:
  -h --help               Show this screen.
  -f --file=<FILENAME>    Image to process
  -c --compare=<FILENAME> Image to compare
  -d --distance=<NUMBER>  Maximum RGB distance between matching colors
  -e --cache=<FILENAME>   Cache filename to use
'''

import sys
import os
import numpy
from PIL import Image
from docopt import docopt
from collections import namedtuple
import operator 

import logging
import struct
import math

import colormath.color_objects 
import colormath.color_conversions 
import colormath.color_diff 

import yaml
import hashlib

def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()

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

def rgb_distance_linear(c1, c2):
  '''
  This is a fast&dirty filter which is good for colors which are slighy off because of the problems
  with the screenshots, or colors modified intentionally
  '''
  return abs(c1[0]-c2[0])+abs(c1[1]-c2[1])+abs(c1[2]-c2[2])

def matching_colors_linear(c1, c2, max_distance):
  return rgb_distance_linear(c1, c2) < max_distance

def rgb_distance(color1, color2):
  '''
  This is the real thing based on http://hanzratech.in/2015/01/16/color-difference-between-2-colors-using-python.html
  It is also slow
  '''
  color1_rgb = colormath.color_objects.sRGBColor(color1[0], color1[1], color1[2])
  color2_rgb = colormath.color_objects.sRGBColor(color2[0], color2[1], color2[2])

  color1_lab = colormath.color_conversions.convert_color(color1_rgb, colormath.color_objects.LabColor)
  color2_lab = colormath.color_conversions.convert_color(color2_rgb, colormath.color_objects.LabColor)
  delta_e = colormath.color_diff.delta_e_cie2000(color1_lab, color2_lab)
  return delta_e

def matching_colors(c1, c2, max_distance):
  return rgb_distance_linear(c1, c2) < max_distance

def find_matching_color(color, palette, max_distance):
  for c in palette.keys():
    if color != c and matching_colors(color, c, max_distance):
      return True, c
  return False, None

def palette(image, max_rgb_distance):
  '''
  Returns a 'palette': dictionary [color](occupied area)
  '''
  palette = {}
  arr = numpy.asarray(image)
  data = asvoid(arr).ravel()
  for point in data:    
    point_color = struct.unpack("BBB", point)
    if point_color in palette.keys(): 
      # I have seen this exact color before. I expect this condition to hit often
      palette[point_color] += 1       
    else:                             
      is_match, matching_color = find_matching_color(point_color, palette, max_rgb_distance)
      if is_match:
        # There is a reasonable match in the collected so far palette
        palette[matching_color] += 1
      else:
        # Something I did not see before
        palette[point_color] = 1

  return len(data), palette


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
  color_palette_sorted = sorted(color_palette.items(), key=operator.itemgetter(1), reverse=True)
  for color, ratio in color_palette_sorted:
     print("{0} {1:.4f}".format(color, 100*ratio))

def palette_distance(color_palette1, color_palette2):
  '''
  Compare two palettes
  Order the palette by color (works better than ordering by area)
  Summ distances between palettes
  '''
  #color_palette_sorted1 = sorted(color_palette1.items(), key=operator.itemgetter(1), reverse=True)
  #color_palette_sorted2 = sorted(color_palette2.items(), key=operator.itemgetter(1), reverse=True)

  color_palette_sorted1 = sorted(color_palette1.keys())
  color_palette_sorted2 = sorted(color_palette2.keys())

  palette_size = min(len(color_palette1), len(color_palette2))
  distance = 0
  for idx in range(palette_size):
    c1 = (color_palette_sorted1[idx])
    c2 = (color_palette_sorted2[idx])
    distance += rgb_distance_linear(c1, c2)
    # size of the area occupied by the color impacts the distance
    # how can I add "area percentage distance" and "RGB distance"?
    # distance += 100*area_weighting*abs(color_palette1[c1]-color_palette2[c2]) 
  distance = (distance/palette_size)
  return distance


PaletteCached = namedtuple('PaletteCached', ['filename', 'distance', 'palette'])

def cache_key(max_distance, file_md5):
  return str(max_distance)+file_md5

def update_cache(cache_filename, filename, palette, max_distance):
  file_md5 = md5sum(filename)
  with open(cache_filename, 'r', newline='') as f:
    cache_data = yaml.load(f)
  
  key = cache_key(max_distance, file_md5)
  if key in cache_data:
    logger.info("File {0} MD5 {1} is in cache".format(filename, file_md5))
    return

  cache_data[key] = PaletteCached(filename, max_distance, palette)
  with open(cache_filename, 'w', newline='') as f:
    f.write(yaml.dump(cache_data))


def load_from_cache(cache_filename, image_file, max_distance):
  if not os.path.exists(cache_filename):
    cache_data = {}
    with open(cache_filename, 'w', newline='') as f:
      f.write(yaml.dump(cache_data))

  with open(cache_filename, 'r', newline='') as f:
    cache_data = yaml.load(f)
  
  file_md5 = md5sum(image_file)
  key = cache_key(max_distance, file_md5)
  if key in cache_data:
    logger.info("File {0} is in cache".format(filename))
    paletteCached = cache_data[key]
    return True, paletteCached.palette
  
  return False, None
    

if __name__ == '__main__':
  arguments = docopt(__doc__, version='0.1')
  logging.basicConfig()    
  logger = logging.getLogger('hamming')
  logger.setLevel(logging.INFO)  
  image_file = arguments['--file']

  cache_filename = arguments['--cache'] 
  if cache_filename is None:
    cache_filename = ".cache.yaml"

  rgb_distance_str = arguments['--distance']
  if rgb_distance_str is None:
    rgb_distance_str = "20"
  rgb_max_distance =  int(rgb_distance_str, 10)

  isInCache, color_palette = load_from_cache(cache_filename, image_file, rgb_max_distance)
  if not isInCache:
    image = Image.open(image_file, 'r').convert('RGB')
    image_size, color_palette = palette(image, rgb_max_distance)
    normalize_color_palette(image_size, color_palette)
    update_cache(cache_filename, image_file, color_palette)

  compare_file = arguments['--compare']
  if compare_file is None:
    print(image_file)
    print_color_palette(color_palette)
    exit(0)

  isInCache, color_palette_compare = load_from_cache(cache_filename, compare_file, rgb_max_distance)
  if not isInCache:
    image = Image.open(compare_file, 'r').convert('RGB')
    image_size, color_palette_compare = palette(image, rgb_max_distance)
    normalize_color_palette(image_size, color_palette_compare)
    update_cache(cache_filename, image_file, color_palette_compare)

  distance = palette_distance(color_palette, color_palette_compare)
  if distance == 0:
    print("{0}, {1} perfect match".format(image_file, compare_file))
  else:
    print("{0}, {1} Distance {2:08.0f}".format(image_file, compare_file, distance))


