# import the necessary packages
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import json
import os
import sys
from skimage.filters import rank
from skimage.morphology import disk
from skimage.segmentation import flood, flood_fill
import floodFill
from pathlib import Path
np.set_printoptions(threshold=sys.maxsize)

NAME = "floodfill_v1_dist20_20__withoutLimit.jpg"

def find_protein(image_path, mask_dir, json_file):
	# construct the argument parse and parse the arguments
	"""ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(ap.parse_args())"""


	# load the image and perform pyramid mean shift filtering
	# to aid the thresholding step
	image = cv2.imread(image_path)
	#cv2.imshow("Input", image)
	#cv2.waitKey()

	# convert the mean shift image to grayscale, then apply
	# Otsu's thresholding
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#gray = rgb_to_grey(shifted, gray)
	#cv2.imshow("gray", gray)
	#cv2.waitKey(0)
	#cv2.imwrite("gray_p10299.jpg", gray)

	#denoise image
	denoised = rank.median(gray, disk(2))
	#cv2.imshow("denoised", denoised)
	#cv2.waitKey(0)

	# flood_fill
	brighten = brighten_image(denoised, json_file)

	#threshold
	thresh = cv2.threshold(brighten, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cv2.imshow("Thresh", thresh)
	cv2.waitKey()
	cv2.imwrite("thresh_p1_0299.jpg", thresh)

	"""shifted = cv2.pyrMeanShiftFiltering(gray, 21, 51)
	cv2.imshow("Pyramid", image)
	cv2.waitKey()
	cv2.imwrite("pyramid_p10299.jpg", shifted)"""


	#clahe
	"""clahe = cv2.createCLAHE(clipLimit = 10)
	c_gray = clahe.apply(gray) + 5
	cv2.imshow("gray clahe", c_gray)
	cv2.waitKey(0)
	cv2.imwrite("clahe_p1_0299.jpg", c_gray)"""


	# compute the exact Euclidean distance from every binary
	# pixel to the nearest zero pixel, then find peaks in this
	# distance map
	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=20,
		labels=thresh)

	# perform a connected component analysis on the local peaks,
	# using 8-connectivity, then appy the Watershed algorithm
	#markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	markers = get_marker(image, json_file)

	# local gradient (disk(2) is used to keep edges thin)
	"""gradient = rank.gradient(denoised, disk(5))
	cv2.imshow("gradient", gradient)
	cv2.waitKey(0)
	cv2.imwrite("gradient" + NAME, gradient)"""

	labels = watershed(-D, markers, mask=thresh)
	print("[INFO] {} uniqueD segments found".format(len(np.unique(labels)) - 1))

	# loop over the unique labels returned by the Watershed
	# algorithm
	for i, label in enumerate(np.unique(labels)):

		# if the label is zero, we are examining the 'background'
		# so simply ignore it
		if label == 0:
			continue

		# otherwise, allocate memory for the label region and draw
		# it on the mask
		mask = np.zeros(gray.shape, dtype="uint8")
		mask[labels == label] = 255
		#cv2.imwrite(r"D:\Pycharm_project\chromosome_segmentation\masks\12655_0299_1"+ os.sep +"mask" + str(i) + ".jpg", mask)
		#cv2.imwrite(mask_dir + os.sep + "mask" + str(i) + ".jpg", mask)

		#cv2.imshow("mask", mask)
		#cv2.waitKey()
		# detect contours in the mask and grab the largest one
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)

		# draw a circle enclosing the object
		((x, y), r) = cv2.minEnclosingCircle(c)
		#cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 1)
		#cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		#draw contours
		cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)
	name = NAME
	cv2.imwrite(name, image)

	path = r"D:\Pycharm_project\chromosome_segmentation\buffer\brigh" + os.sep + name
	draw_dots(path, json_file)

def draw_dots(img_path, json_file):
	#path = r"D:\Pycharm_project\chromosome_segmentation\SHIDC-B-Ki-67\256x256_cropped_images\test256\12655_0299_1.json"
	path = json_file
	image = cv2.imread(img_path)
	with open(path, "r") as f:
		datas = json.load(f)
		for data in datas:
			cv2.circle(image, (data['x'], data['y']), radius=0, color=(0, 0, 255), thickness=4)
		cv2.imshow("dots", image)
		cv2.waitKey(0)
		cv2.imwrite(img_path[:-4] + "_contoured.jpg", image)

def rgb_to_grey(img, gray):
	r_img = []
	for i, row in enumerate(img):
		new_row = []
		for y, pixel in enumerate(row):
			new_pixel = int((pixel[0] + pixel[1] + pixel[2] * 2)/4)
			new_row.append(new_pixel)
			gray[i,y] = new_pixel
		r_img.append(new_row)
	return gray

def get_marker(img,json_file):
	path = json_file
	with open(path, "r") as f:
		datas = json.load(f)
		ind = []
		for data in datas:
			ind.append([data['x'], data['y']])
	res = np.zeros([np.shape(img)[0], np.shape(img)[1]])
	print(len(ind))
	for i, cor in enumerate(ind):
		res[cor[1], cor[0]] = i
	return res

def brighten_image(img, json_file):
    new_img = img.copy()
    with open(json_file, "r") as f:
        datas = json.load(f)
        for data in datas:
            #new_img = flood_fill(new_img, (0, 0), 0, tolerance=10)
            new_img = floodFill.floodFill(new_img, np.shape(new_img)[0], np.shape(new_img)[1], data['y'], data['x'], new_img[data['y'], data['x']], 0, 10, 8000)
            #new_img = flood_fill(new_img, (data['y'], data['x']), 0, tolerance=10)
            #cv2.imshow("brightened", new_img)
            #cv2.waitKey()
    cv2.imshow("brightened", new_img)
    cv2.waitKey()
    cv2.imwrite("brightened_20_800.jpg", new_img)
    return new_img


if __name__ == '__main__':

	find_protein(r"D:\Pycharm_project\chromosome_segmentation\SHIDC-B-Ki-67\bare_images\Test\p1_0299_1.jpg" ,r"D:\Pycharm_project\chromosome_segmentation\buffer\mask", r"D:\Pycharm_project\chromosome_segmentation\SHIDC-B-Ki-67\bare_images\Test\p1_0299_1.json")
    #draw_dots("D:\Pycharm_project\chromosome_segmentation\watershed_example\min_dist4_c5.jpg")