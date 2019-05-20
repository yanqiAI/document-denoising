# coding:utf-8
'''
test original size image;
2019/04/23 by yanqi
1. clip test images
2. test
3. merge results
'''
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from model import get_model
from IPython.core.debugger import Tracer

def clip_one_test(data_dir):
    img = cv2.imread(data_dir)
    img_h, img_w = img.shape[:2]

    # clip m*n
    m = img_h // 256
    n = img_w // 256
    gh = m * 256
    gw = n * 256
    img_resize = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_LINEAR)

    # processing
    gx, gy = np.meshgrid(np.linspace(0, gw, n + 1), np.linspace(0, gh, m + 1))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)

    divide_image = np.zeros([m, n, 256, 256, 3], np.uint8)

    clip_images = []
    for i in range(m):
        for j in range(n):
            divide_image[i, j, ...] = img_resize[gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1]]
            clip_images.append(divide_image[i, j, ...])
    print('{} has cliped {} images successfully!'.format(data_dir, len(clip_images)))
    return clip_images, m, n

def merge_cliped_data(cliped_images, m, n, img_size = 256):
    '''
    :param cliped_images: cliped list
    :param m: the number of rows
    :param n: the number of cols
    :param img_size: cliped image size (default=256*256)
    :return: dst: merged image
    '''
    gh = m * img_size
    gw = n * img_size

    gx, gy = np.meshgrid(np.linspace(0, gw, n + 1), np.linspace(0, gh, m + 1))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)
    dst = np.zeros((gh, gw, 3), np.uint8)

    count = 0
    for i in range(m):
        for j in range(n):
            dst[gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1]] = cliped_images[count]
            count += 1

    return dst

def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)

def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default='E:/datasets/denoising/processing_data/blur_test',
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, default='checkpoints/weights.059-1.705-30.58344.hdf5',
                        help="trained weight file")
    parser.add_argument("--output_dir", type=str, default='result/0429',
                        help="if set, save resulting images")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    model = get_model(args.model)
    model.load_weights(weight_file)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:
        #Tracer()()
        image = cv2.imread(str(image_path)) # read original image
        img_h, img_w, _ = image.shape

        # clip original image
        cliped_image, m, n = clip_one_test(str(image_path))

        # prediction for cliped images
        print('satrt to predict {} image!'.format(image_path.name))

        denoised = []
        for img in cliped_image:
            pred = model.predict(np.expand_dims(img, 0))
            channels = pred.shape[-1]
            if channels == 6:
                # pred from 6 channels transform to 3 channels
                pred_fg = pred[:, :, :, :3]
                pred_bg = pred[:, :, :, 3:]
                pred = pred_fg + pred_bg
            denoised_image = get_image(pred[0])
            denoised.append(denoised_image)

        # merge test results
        dst = merge_cliped_data(denoised, m, n, img_size=256)

        # resize dst
        dst = cv2.resize(dst, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        print('{} has merged successfully!'.format(image_path.name))

        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".jpg", dst)
        else:
            cv2.imshow("result", out_image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0

if __name__ == '__main__':
    main()