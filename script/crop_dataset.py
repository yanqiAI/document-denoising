# coding:utf-8
import os
import sys
import cv2
import numpy as np
from IPython.core.debugger import Tracer

'''
切分图像，得到一系列的图像块，生成训练样本
1）对边界先做切除；
2）图像切块256*256大小
'''
def generate_train_dataset(data_path, save_path, img_size=256):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    files_list = os.listdir(data_path)

    for file in files_list:
        file_path = os.path.join(data_path, file)
        image = cv2.imread(file_path, 0)

        #image = image[450:-350, 300:-300] #切除空白边界
        #cv2.imshow('image_clip', image_clip)
        h, w = image.shape[:2]

        # 切块成 m*n 个
        m = int(h / img_size)
        n = int(w / img_size)

        gh = m * img_size
        gw = n * img_size

        image_clip_resize = cv2.resize(image, (gw, gh), interpolation=cv2.INTER_LINEAR)
        gx, gy = np.meshgrid(np.linspace(0, gw, n + 1), np.linspace(0, gh, m + 1))
        gx = gx.astype(np.int)
        gy = gy.astype(np.int)

        divide_image = np.zeros([m, n, img_size, img_size], np.uint8)
        #Tracer()()
        count = 0
        for i in range(m):
            for j in range(n):
                divide_image[i, j, ...] = image_clip_resize[gy[i][j] : gy[i + 1][j + 1], gx[i][j] : gx[i + 1][j + 1]]
                count += 1
                cv2.imwrite(save_path + '/' + file[:-4] + '_' + str('%04d'%count) + '.jpg', divide_image[i, j, ...])
                #Tracer()()
        print('{} has been cliped successfully!'.format(file))
    print('_' * 50)
    print('All files have been cliped successfully!')

if __name__ == '__main__':
    blur_path = 'E:/datasets/denoising/processing_data/background'
    clear_path = 'E:/datasets/denoising/dirtuy_office_process/train_cleaned'

    blur_save_path = 'E:/datasets/denoising/processing_data/background_clip'
    clear_save_path = 'E:/datasets/denoising/dirtuy_office_process/train_cleaned_clip'

    generate_train_dataset(blur_path, blur_save_path, img_size=256)
    #generate_train_dataset(clear_path, clear_save_path, img_size=256)