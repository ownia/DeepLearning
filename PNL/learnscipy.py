import skimage.transform as trans
import imageio as im
import numpy as np
from scipy.spatial.distance import pdist, squareform

img = im.imread('brownbear.jpg')
print(img.dtype, img.shape)
img_tinted = img * [1, 0.95, 0.9]
img_tinted = trans.resize(img_tinted, (250, 200))
im.imsave('brownbear_tinted.jpg', img_tinted)

