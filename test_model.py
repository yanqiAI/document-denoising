import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model
import time
from IPython.core.debugger import Tracer

def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default='E:/datasets/denoising/processing_data/blur_cliped_test',
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, default='checkpoints/weights.040-3.169-29.39941-0.94307.hdf5',
                        help="trained weight file")
    parser.add_argument("--output_dir", type=str, default='E:/datasets/denoising/results/n2c/0517_127_2',
                        help="if set, save resulting images")
    args = parser.parse_args()
    return args

def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)

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
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape
        image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
        h, w, _ = image.shape

        out_image = np.zeros((h, w * 4, 3), dtype=np.uint8)
        start_time = time.time()
        pred = model.predict(np.expand_dims(image, 0))
        end_time = time.time()
        cost_time = end_time - start_time
        print('this image has {} s'.format(cost_time))

        channels = pred.shape[-1]
        if channels == 6:
            # pred from 6 channels transform to 3 channels
            pred_fg = pred[:, :, :, :3]
            pred_bg = pred[:, :, :, 3:]
            pred = pred_fg + pred_bg

        denoised_fg = get_image(pred_fg[0])
        denoised_bg = get_image(pred_bg[0])
        denoised_image = get_image(pred[0])
        out_image[:, :w] = image
        out_image[:, w:w * 2] = denoised_fg
        out_image[:, w * 2:w * 3] = denoised_bg
        out_image[:, w * 3:] = denoised_image

        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".jpg", out_image)
            print('{} has saved successfully!'.format(image_path.name))
        else:
            cv2.imshow("result", out_image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0

if __name__ == '__main__':
    main()