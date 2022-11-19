import cv2 as cv
import os
import math
import csv
import numpy as np
import pandas as pd
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

"""Bu kod verilien yolo koordinatlari uzerinden kromozomlari keser"""

def main():
    cut_as_patch()
    #img = cv.imread(r"D:\Pycharm_project\chromosome_segmentation\test\FATMA_09.jpg")
    #print(img)
    #seperate_all_images(r"D:\Pycharm_project\chromosome_segmentation\train\WithChromoLabels_noenhance-20220716T190737Z-001\WithChromoLabels_noenhance\test", 3454, r"D:\Pycharm_project\chromosome_segmentation\todelete")
    #seperate_images(r"D:\Pycharm_project\chromosome_segmentation\train\BRC_AE_01_03.jpg", r"D:\Pycharm_project\chromosome_segmentation\train\BRC_AE_01_03.txt", r"D:\Pycharm_project\chromosome_segmentation\cuted_image")
    #seperate_images(r"D:\Pycharm_project\chromosome_segmentation\train\BRC_AE_01_06.jpg",
                    #r"D:\Pycharm_project\chromosome_segmentation\train\BRC_AE_01_06.txt",
                    #r"D:\Pycharm_project\chromosome_segmentation\cuted_image",
                    #"hello.jpg")

def cut_as_patch():
    file_list = os.listdir("D:\Pycharm_project\chromosome_segmentation\images")

    for i, file in enumerate(file_list):
        img = cv.imread("D:\Pycharm_project\chromosome_segmentation\images"+ os.sep + file)
        res = img[0:256, 0:256]
        print(res.shape)
        cv.imwrite(file, res)
        #res = cv.resize(img, dsize=(256, 256), interpolation=cv.INTER_AREA)
        #cv.imwrite(file, res)

def seperate_images(img_path, text_path, image_name, writer):
    """
    :param img_path: absolute path of image
    :param text_path: absolute path of text (text formated as yolov5 standarts)
    :return: NONE
    """
    #read image
    img = cv.imread(img_path)
    #get image width and length
    img_length = img.shape[0]
    img_width = img.shape[1]
    #print(img_length, img_width)
    cor_list = get_coordinates(text_path, img_length, img_width)

    #write label texts
    i = 0
    for row in cor_list[1]:
        words = row.split()
        #my_file = open(final_path + os.sep + image_name[:-4] + "_" + str(i) + ".txt", "w")
        #write label
        csv_line = [image_name, words[0]]
        #my_file.write(words[0])

        writer.writerow({'path': image_name[:-4] + "_" + str(i) + ".jpg", 'label': str(words[0])})
        i+=1

    i = 0
    #create new cutted images
    for row in cor_list[0]:
        crop_img = img[row[2]:row[2] + row[4], row[1]:row[1] + row[3]]
        #show image
        #cv.imshow("cropped", crop_img)
        #cv.waitKey(0)
        #write_image

        cv.imwrite(image_name[:-4] + "_" + str(i) + ".jpg", crop_img)

        #resize the image

        img2 = cv.imread(image_name[:-4] + "_" + str(i) + ".jpg")
        res = cv.resize(img2, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
        #write the new image
        cv.imwrite(image_name[:-4] + "_" + str(i) + ".jpg", res)

        i+=1


def get_coordinates(path, length, width):
    """
    from a yolov5 standart text file, takes coordinate of all objects
    :return : a list of list containing coordinates
    """
    #open file
    my_file = open(path)
    #read lines
    string_list = my_file.readlines()
    #return value
    ret = []
    #get every coordinates for every row
    for row in string_list:
        #split every element in a row
        words = row.split()
        #rate yolov5 coordinate system to np array index
        rated = rate(words, length, width)
        ret.append(rated)

    return ret, string_list

def rate(coordinate_list, length, width):
    """
    :param coordinate_list: yolov5 standart line
    :return: a list with numpy indexable coordinates
    """
    #return value
    cor = []

    #add type
    cor.append(coordinate_list[0])
    #get object box length and with
    o_width = math.floor(float(coordinate_list[3])*width)
    o_length = math.floor(float(coordinate_list[4])*length)
    #add starting X coordinate
    cor.append(math.floor(float(coordinate_list[1])*width) - math.floor(o_width/2))
    #add starting Y coordinate
    cor.append(math.floor(float(coordinate_list[2])*length) - math.floor(o_length / 2))
    #add width and length
    cor.append(o_width)
    cor.append(o_length)
    return cor

def seperate_all_images(path_of_dir, datasize, final_path):
    """
    :param: datasize determines how many picture do you want to cut
    :return: none seperates all images in a folder
    """
    i = 0
    image_name = ""
    #create a csv file to load label and image path
    filed_names = ['path', 'label']
    csv_file = open(final_path + os.sep + "z_labels.csv", "w")
    writer = csv.DictWriter(csv_file, fieldnames=filed_names)

    for files in os.listdir(path_of_dir):
        if files.endswith("jpg"):
            image_name = files
        elif files.endswith("txt"):
            #print(final_path)
            seperate_images(path_of_dir + os.sep + image_name, path_of_dir + os.sep + files, image_name, writer)
            i += 1
            print(i)
        if i >= datasize:
            break
    csv_file.close()
    read_file = pd.read_csv(final_path + os.sep + "z_labels.csv")
    read_file.to_excel(final_path + os.sep + "z_labels.xlsx", index=None, header=True)


if __name__ == '__main__':
    main()






