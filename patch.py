import cv2
import math
import numpy as np

from pathlib import Path

file_name = "cameraman.tif"
save_path = './img_patch'  # create dir sunxu
Path(save_path).mkdir(parents=True, exist_ok=True)

# 分块大小
height = 64
width = 64
length = 64

# 重叠部分大小
over_x = 10
over_y = 10
over_z = 10
h_val = height - over_x
w_val = width - over_y
l_val = length - over_z

# Set whether to discard an image that does not meet the size
mandatory = False

img = cv2.imread(file_name)

print(img.shape)
# original image size
original_height = img.shape[0]
original_width = img.shape[1]
original_length = img.shape[2]

max_row = float((original_height - height) / h_val) + 1
max_col = float((original_width - width) / w_val) + 1
max_len = float((original_length - length) / l_val) + 1

# block number
max_row = math.ceil(max_row) if mandatory == False else math.floor(max_row)
max_col = math.ceil(max_col) if mandatory == False else math.floor(max_col)
max_len = math.ceil(max_len) if mandatory == False else math.floor(max_len)

print(max_row)
print(max_col)
print(max_len)

images = []
for k in range(max_len):
    for i in range(max_row):
        images_temp = []
        for j in range(max_col):
            temp_path = save_path + '/' + str(i) + '_' + str(j) + '_'
            if ((width + j * w_val) > original_width and (i * h_val + height) <= original_height):  # Judge the right most incomplete part
                temp = img[i * h_val:i * h_val + height, j * w_val:original_width, :]
                temp_path = temp_path + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.jpg'
                cv2.imwrite(temp_path, temp)
                images_temp.append(temp)
            elif ((height + i * h_val) > original_height and (j * w_val + width) <= original_width):  # Judge the incomplete part at the bottom
                temp = img[i * h_val:original_height, j * w_val:j * w_val + width, :]
                temp_path = temp_path + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.jpg'
                cv2.imwrite(temp_path, temp)
                images_temp.append(temp)
            elif ((width + j * w_val) > original_width and (i * h_val + height) > original_height):  # Judge the last slide
                temp = img[i * h_val:original_height, j * w_val:original_width, :]
                temp_path = temp_path + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.jpg'
                cv2.imwrite(temp_path, temp)
                images_temp.append(temp)
            else:
                temp = img[i * h_val:i * h_val + height, j * w_val:j * w_val + width, :]
                temp_path = temp_path + str(temp.shape[0]) + '_' + str(temp.shape[1]) + '.jpg'
                cv2.imwrite(temp_path, temp)
                images_temp.append(temp)  # The rest of the complete

        images.append(images_temp)

print(len(images))