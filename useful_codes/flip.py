import cv2 as cv
import numpy as np
import os
"""
This files flips images and their labels
"""

def main():
    path = r"D:\Pycharm_project\chromosome_segmentation\train2"
    flip(path)


def flip(root):
    for files in os.listdir(root):
        print(files)
        if files.endswith("jpg"):
            flip_img(root, files)
        elif files.endswith("txt"):
            flip_text(root, files)

def flip_img(root, name):
    src = cv.imread(root + os.sep + name)
    flipped = cv.flip(src, 1)
    cv.imwrite(name[:-4] + "_flipped.jpg", flipped)


def flip_text(root, name):
    #read file
    with open(root + os.sep + name) as myfile:
        lines = myfile.readlines()

    #flip values i.e. 1- value
    with open(name[:-4] + "_flipped.txt", 'a') as new_text:
        for row in lines:
            words = row.split()
            new_line = ""
            for i, word in enumerate(words):
                if i == 1:
                    new_word = 1 - float(word)
                    new_line += str(new_word) + " "
                elif i == 0 or i == 2 or i == 3:
                    new_line += word + " "
                else:
                    new_line += word + "\n"
            new_text.write(new_line)


if __name__ == '__main__':
    main()