import cv2
import os
import glob
from xml.etree import ElementTree

xml_all_files = os.listdir('xml')
img_all_files = os.listdir('image')


def next_index_in_folder(folder_name):
    if not os.listdir(folder_name):
        return 1
    else:
        list_of_files = glob.glob(folder_name + '/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        file_index = latest_file.split('\\')[1].split('.')[0]
        return int(file_index) + 1


for x in range(len(xml_all_files)):
    xml_file_name = xml_all_files[x]
    xml_full_file = os.path.abspath(os.path.join('xml', xml_file_name))

    img_file_name = img_all_files[x]
    img_full_file = os.path.abspath(os.path.join('image', img_file_name))

    dom = ElementTree.parse(xml_full_file)
    cells = dom.findall('object')
    cell_pos = []

    print("Reading.." + xml_full_file)
    print("Reading.." + img_full_file)
    print("#########################")

    if xml_file_name.split('.')[0] == img_file_name.split('.')[0]:

        for c in cells:
            cell_type = c.find('name').text
            cell_location = c.find('bndbox')

            x_min = cell_location.find('xmin').text
            y_min = cell_location.find('ymin').text
            x_max = cell_location.find('xmax').text
            y_max = cell_location.find('ymax').text

            cell_pos.append([[cell_type], [int(x_min), int(y_min), int(x_max), int(y_max)]])

        cell_img = cv2.imread(img_full_file)

        for c in cell_pos:
            crop_img_save_directory = 'train/' + c[0][0]
            crop_cell_img = cell_img[c[1][1]:c[1][3], c[1][0]:c[1][2]]

            if not os.path.exists(crop_img_save_directory):
                os.makedirs(crop_img_save_directory)

            if not (crop_cell_img is None):
                cv2.imwrite(crop_img_save_directory + "/" + str(next_index_in_folder(crop_img_save_directory)) + ".jpg",
                            crop_cell_img)

    cells.clear()
    cell_pos.clear()
