# image-mathching

The scripts target phishing WEB sites and, potentially, phishing e-mails. 

The color-palette script collects colors from the given image(s), compare the colo palettes, return "distance" between the palettes.
This is a reasonably fast way to match screenshots of the WEB sites, flag suspicious phishing sites. The script can be a part of 
the lab Grzegorz Jakowicz is building.

Script text-detection.py collects text boxes from the image, generate a "collage". The idea is that color palette 
matching will work better for areas containing text (it does). Fuzzy hashes like ssdeep can perform better as well when running only for areas in the image containing text. 

 The pipeline handling the screenshots can start with a fuzzy hashes base solution like ssdeep. The fuzzy hash step can be done really fast. A Go application calculate 50M/s 256 bits hamming distances (https://confluence.corp.cyren.com/display/AR/Hamming+distance). If ssdeep does not find a match (negative) try the color palette approach (reasonably fast). If there is a match I would run OCR. We already run EAST in the previous step, discovered text boxes coordinates. We need only Kraken to OCR. 

Links:

* Based on https://stackoverflow.com/questions/18801218/build-a-color-palette-from-image-url
* See also https://stackoverflow.com/questions/75891/algorithm-for-finding-similar-images/83486#83486 - lot of tips here 
* https://stackoverflow.com/questions/1704793/find-images-with-similar-color-palette-with-python
* https://github.com/myint/perceptualdiff
* https://stackoverflow.com/questions/5392061/algorithm-to-check-similarity-of-colors - distance between colors
* http://hanzratech.in/2015/01/16/color-difference-between-2-colors-using-python.html - distance between RGB colors
* https://s3-us-west-2.amazonaws.com/static.pyimagesearch.com/opencv-text-detection/opencv-text-detection.zip - example of working with EAST 
* https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/ - example of working with EAST 