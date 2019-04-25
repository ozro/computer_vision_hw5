import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    # Get coordinates of bbox centers as (y, x, h, w)
    max_h = np.max(bboxes[:,2] - bboxes[:,0])
    coords = [((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2, (bbox[2] - bbox[0]), (bbox[3] - bbox[1])) for bbox in bboxes]
    coords = sorted(coords, key=lambda x: x[0]) # Sort by y coordinate

    prev_coord = coords[0]
    prev_row = 0
    row = []
    rows = []
    rows.append(row)
    for coord in coords:
        if(coord[0] - prev_coord[0] > max_h * 0.75): # moving on to new row
            prev_row += 1
            prev_coord = coord
            row = []
            row.append(coord)
            rows.append(row)
        else: #within previous row
            rows[prev_row].append(coord)
            rows[prev_row] = sorted(rows[prev_row], key=lambda x: x[1]) # maintain row order in x direction

    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

    dataset = []
    min_pad = 10

    for row in rows:
        data = []
        for y,x,h,w in row:
            patch = bw[y-h//2:y+h//2, x-w//2:x+w//2]
            # Pad the patch to a square, using larger dimension
            if h>w:
                pad_y = min_pad
                pad_x = pad_y + h-w
            elif h<w:
                pad_x = min_pad
                pad_y = pad_x + w-h
            else:
                pad_x = 0
                pad_y = 0
            patch = np.pad(patch, ((pad_y, pad_y), (pad_x, pad_x)), 'constant', constant_values=(1, 1))
            patch = skimage.transform.resize(patch, (32, 32))
            patch = skimage.morphology.erosion(patch, skimage.morphology.disk(2))
            # plt.imshow(patch)
            # plt.show()
            patch = np.transpose(patch)
            data.append(patch.flatten())
        dataset.append(data)

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    res = ""
    for data in dataset:
        h1 = forward(data, params, "layer1", sigmoid)
        probs = forward(h1, params, "output", softmax)
        pred_ind = np.argmax(probs, axis = 1)
        res += "".join(letters[pred_ind]) + "\n"

    print(res)
    
