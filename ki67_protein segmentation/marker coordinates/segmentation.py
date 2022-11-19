import tifffile
import numpy as np
import cv2 as cv
from skimage import filters
from skimage.morphology import disk
from skimage import segmentation
from skimage import measure
from skimage import morphology
from scipy import ndimage as ndi
from watershed_file_2 import get_marker, draw_dots
import imutils
import napari
import os

NAME = "deneme2_kernel10.jpg"

spacing = np.array([0.29, 0.26, 0.26])

#read data
image = cv.imread(r"D:\Pycharm_project\chromosome_segmentation\SHIDC-B-Ki-67\bare_images\Test\p1_0299_1.jpg")
viewer = napari.view_image(image, rgb=True)
"""print("shape: {}".format(imagei.shape))
print("dtype: {}".format(imagei.dtype))
print("range: ({}, {})".format(np.min(imagei), np.max(imagei)))"""

#gray
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#edges
edges = filters.scharr(gray)

viewer.add_image(
    edges,
    blending='additive',
    colormap='magenta',
)

#cv.imshow("edge", edges)
#cv.waitKey(0)

#denoise
denoised = filters.rank.median(gray, disk(2))
#denoised = ndi.median_filter(edges)
#cv.imshow("denoised", denoised)
#cv.waitKey(0)

#threshold
thresh = cv.threshold(denoised, 0, 255,
		cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]


viewer.add_image(
    thresh,
    opacity=0.3,
)

#remove_holes
width = 20

remove_holes = morphology.remove_small_holes(
    thresh, width ** 2
)

remove_objects = morphology.remove_small_objects(
    remove_holes, width ** 2
)

viewer.add_image(
    remove_objects,
    name='cleaned',
    opacity=0.3,
);


labels = measure.label(remove_objects)

viewer.add_labels(
    labels,
    opacity=0.5,
)


#segmentation
#add points
transformed = ndi.distance_transform_edt(remove_objects)

maxima = morphology.local_maxima(transformed)
points = viewer.add_points(
    np.transpose(np.nonzero(maxima)),
    name='points',
    size=4
)


"""labels = measure.label(thresh)
labels = labels * 255/305
cv.imshow("labels", labels)
cv.waitKey(0)"""

#get markers
marker_locations = get_marker(image, r"D:\Pycharm_project\chromosome_segmentation\SHIDC-B-Ki-67\bare_images\Test\p1_0299_1.json")
#print(np.transpose(np.nonzero(marker_locations)))
#marker_locations = np.transpose(np.nonzero(marker_locations))
#marker_locations = points.data

#markers = np.zeros(image.shape, dtype=np.uint32)
#marker_indices = tuple(np.round(marker_locations).astype(int).T)
#markers[marker_indices] = np.arange(len(marker_locations)) + 1
#markers_big = morphology.dilation(marker_locations, morphology.ball(5))
kernel = np.ones((5, 5), np.uint8)
markers_big = cv.dilate(marker_locations, kernel, iterations=1)

segmented = segmentation.watershed(
    edges,
    markers_big,
    mask=remove_objects,
)

viewer.add_labels(
    segmented,
)
cv.imshow("threshold", thresh)
cv.waitKey(0)

# loop over the unique labels returned by the Watershed
	# algorithm
for i, label in enumerate(np.unique(segmented)):

    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue

    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[segmented == label] = 255
    #cv2.imwrite(r"D:\Pycharm_project\chromosome_segmentation\masks\12655_0299_1"+ os.sep +"mask" + str(i) + ".jpg", mask)
    #cv2.imwrite(mask_dir + os.sep + "mask" + str(i) + ".jpg", mask)

    #cv2.imshow("mask", mask)
    #cv2.waitKey()
    # detect contours in the mask and grab the largest one
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)

    # draw a circle enclosing the object
    ((x, y), r) = cv.minEnclosingCircle(c)
    #cv.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 1)
    #cv.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #draw contours
    cv.drawContours(image, cnts, -1, (0, 255, 0), 3)
# show the output image
cv.imshow("Output", image)
cv.waitKey(0)
cv.drawContours(image, cnts, -1, (0, 255, 0), 3)

name = NAME
path = r"D:\Pycharm_project\chromosome_segmentation\buffer\segmentation" + os.sep + name
cv.imwrite(path, image)
draw_dots(path,
          r"D:\Pycharm_project\chromosome_segmentation\SHIDC-B-Ki-67\bare_images\Test\p1_0299_1.json")
