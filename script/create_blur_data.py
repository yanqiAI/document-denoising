# coding:utf-8
'''
2019/04/23 by yanqi
create blur document images
'''
import glob
import os
import cv2
import random
from IPython.core.debugger import Tracer

def blur_data(clear_path, background_path, save_blur_path):
    # create save blur images folder
    if not os.path.exists(save_blur_path):
        os.mkdir(save_blur_path)

    # file list
    clear_images = sorted(glob.glob(os.path.join(clear_path, '*.JPEG')))
    clear_images += sorted(glob.glob(os.path.join(clear_path, '*.jpg')))
    clear_images += sorted(glob.glob(os.path.join(clear_path, '*.png')))

    bg_images = sorted(glob.glob(os.path.join(background_path, '*.JPEG')))
    bg_images += sorted(glob.glob(os.path.join(background_path, '*.jpg')))
    bg_images += sorted(glob.glob(os.path.join(background_path, '*.png')))

    # processing every clear image, random choose background image
    for clear_img in clear_images:
        clear_image = cv2.imread(clear_img)
        clear_image_gray = cv2.cvtColor(clear_image, cv2.COLOR_BGR2GRAY)  # RGB2GRAY

        # 二值化处理
        ret, clear_image_gray_binary = cv2.threshold(clear_image_gray, 127, 1, cv2.THRESH_BINARY)
        h, w = clear_image_gray.shape[:2]

        # random choose one background image
        bg_img = ''.join(random.sample(bg_images, 1))
        bg_image = cv2.imread(bg_img)
        bg_image_gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY) # RGB2GRAY

        # processing blur image
        bg_image_gray_resize = cv2.resize(bg_image_gray, (w, h), interpolation=cv2.INTER_LINEAR)
        blur_image_gray = cv2.multiply(clear_image_gray_binary, bg_image_gray_resize)

        # save images
        clear_gray_path = clear_path.replace('clear_train', 'clear_gray_train')
        if not os.path.exists(clear_gray_path):
            os.makedirs(clear_gray_path)

        #Tracer()()
        cv2.imwrite(save_blur_path + '/' + clear_img[-9:], blur_image_gray) # blur image ====> gray image
        cv2.imwrite(clear_gray_path + '/' + clear_img[-9:], clear_image_gray) # clear image ====> gray image
        #Tracer()()
        print('{} files have processed successfully!'.format(clear_img[:-4]))
    print('all files have processed successfully!')

if __name__ == '__main__':
    # data path
    clear = 'E:/datasets/denoising/processing_data/clear_train'
    background = 'E:/datasets/denoising/processing_data/background'
    processing = 'E:/datasets/denoising/processing_data/blur_gray_train'

    blur_data(clear, background, processing)