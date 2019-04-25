import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # Denoise
    image = skimage.restoration.denoise_bilateral(image, multichannel=True)
    # Greyscale
    image = skimage.color.rgb2gray(image)
    # Threshold the image
    th = skimage.filters.threshold_otsu(image)
    bw = image < th
    # Morphology
    bw = skimage.morphology.closing(bw, skimage.morphology.disk(3))
    bw = skimage.segmentation.clear_border(bw)
    # Label
    label = skimage.morphology.label(bw, connectivity=2)
    props = skimage.measure.regionprops(label)
    # Get large boxes
    median = np.median([x.area for x in props])
    th = median / 2 # Get boxes that are larger than half the mean size
    bboxes = np.asarray([prop.bbox for prop in props if prop.area > th])
    bw = (1-bw).astype(np.float)

    return bboxes, bw
