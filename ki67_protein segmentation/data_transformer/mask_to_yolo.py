import cv2 as cv
import os
import numpy as np
import json
"""
this program transfers code to yolo format
"""

def get_class(img_path, json_file):
    """
    takes a mask image and a json file containing coordinates. If finds a point intersection
    with the mask returns the class in the json file. else returns -1
    """
    img = cv.imread(img_path)
    p_class = -1
    with open(json_file,"r") as f:
        datas = json.load(f)
        for data in datas:
            if set(img[data["x"], data["y"]]) == set([255,255,255]):
                p_class = data["label_id"]
    return p_class
    """for file in os.listdir(dir_path):
        img = cv.imread(dir_path + os.sep + file)
        if(set(img[x,y]) == set((255,255,255))):
            print(img[x,y])
            cv.imshow("hello",img)
            cv.waitKey(0)"""


def get_boundary_coordinates(image_path):
    """
    takes a mask image
    returns boundary box coordinates
    """
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(contours[0])
    cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #cv.imshow("hello", contours)
    #cv.waitKey(0)
    return str((x + w/2)/np.shape(image)[0]) + " " + str((y + h/2)/np.shape(image)[0]) + " " + str(w/np.shape(image)[0])+ " " + str(h/np.shape(image)[0])

def get_yolo_line(img_path, json_path):
    line = []
    p_class = get_class(img_path, json_path)
    if p_class != -1:
        line.append(str(p_class) + " " + get_boundary_coordinates(img_path))
    return line

def create_yolo_file(masks_dir, json_file_path, file_name, save_path):

    with open(save_path + os.sep + file_name[:-5] + ".txt", "w") as f:
        for file in os.listdir(masks_dir):
            line = get_yolo_line(masks_dir + os.sep + file, json_file_path)
            #if there is an intersection of coordinate and mask
            if len(line) > 0:
                f.write(line[0])
                f.write("\n")

def draw_dots(img, json_file):
	#path = r"D:\Pycharm_project\chromosome_segmentation\SHIDC-B-Ki-67\256x256_cropped_images\test256\12655_0299_1.json"
	path = json_file
	with open(path, "r") as f:
		datas = json.load(f)
		for data in datas:
			cv.circle(img, (data['x'], data['y']), radius=0, color=(0, 0, 255), thickness=1)
		cv.imshow("dots", img)
		cv.waitKey(0)
		#cv.imwrite(img_path[:-4] + "_circled.jpg", image)

if __name__ == '__main__':
    #get_boundary_coordinates()
    #get_class(r"D:\Pycharm_project\chromosome_segmentation\masks\mask3.jpg", r"D:\staj\yollanacaklar\SHIDC-B-Ki-67_dataset_results\12655_0299_1.json")
    #print(get_yolo_line(r"D:\Pycharm_project\chromosome_segmentation\masks\mask3.jpg", r"D:\staj\yollanacaklar\SHIDC-B-Ki-67_dataset_results\12655_0299_1.json"))
    create_yolo_file(r"D:\Pycharm_project\chromosome_segmentation\masks2\p10_0035_9", r"D:\Pycharm_project\chromosome_segmentation\SHIDC-B-Ki-67\bare_images\Test\p10_0035_9.json", "p10_0035_9.json",r"D:\Pycharm_project\chromosome_segmentation\buffer")
