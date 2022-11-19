import os

"""bu program yolo labellarinda ki siniflari 1 sayi azaltir. (yolo labellari 0 dan basladigi icin)"""

def main():
    path_of_dir = r"D:\staj\mmdetection\WithChromoLabels_noenhance-20220729T090856Z-001\WithChromoLabels_noenhance\test"
    ext = "txt"

    for files in os.listdir(path_of_dir):
        if files.endswith(ext):
            change_labels_of_file(path_of_dir + os.sep + files)

    #change_labels_of_file(path)


def change_labels_of_file(path):
    my_file = open(path)
    string_list = my_file.readlines()

    my_file.close()

    my_file = open(path, "w")
    my_file.write("")
    my_file.close()

    my_file = open(path, "a")

    for row in string_list:
        words = row.split()
        #decrease the label by 1
        new_label = int(words[0]) - 1
        words[0] = str(new_label)
        #the new list to string
        new_row = ""
        for i in range(len(words) - 1):
            new_row += words[i] + " "
        new_row += words[len(words) - 1]

        #write new content
        my_file.write(new_row + "\n")
    my_file.close()




if __name__ == '__main__':
    main()