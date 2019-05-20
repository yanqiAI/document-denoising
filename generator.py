from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence
from keras import backend as K
from IPython.core.debugger import Tracer

class DeblurImageGenerator(Sequence):
    def __init__(self, blur_image_dir, clear_image_dir, batch_size=32, image_size=64):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.blur_image_paths  = [p for p in Path(blur_image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.clear_image_paths = [p for p in Path(clear_image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.blur_image_num = len(self.blur_image_paths)
        self.clear_image_num = len(self.clear_image_paths)
        self.batch_size = batch_size
        self.image_size = image_size

        if self.blur_image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        assert self.blur_image_num  == self.clear_image_num
        return int(self.blur_image_num // self.batch_size)

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 6), dtype=np.uint8)
        # y_fg = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        # y_bg = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        # clear_mask = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            blur_image_path = random.choice(self.blur_image_paths)
            blur_image = cv2.imread(str(blur_image_path))
            clear_image_path = str(blur_image_path).replace('blur_train_clip', 'clear_train_clip')
            clear_image = cv2.imread(clear_image_path)

            # add clear_mask
            ret, clear_image_mask =  cv2.threshold(clear_image, 127, 1, cv2.THRESH_BINARY)
            clear_image_foreground = clear_image * (1 - clear_image_mask)
            clear_image_background = clear_image * clear_image_mask

            h, w, _ = blur_image.shape

            if h >= image_size and w >= image_size:
                h, w, _ = blur_image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                blur_patch  = blur_image[i:i + image_size, j:j + image_size]


                # add clear_foreground_path and clear_background_path
                clear_foreground_patch = clear_image_foreground[i:i + image_size, j:j + image_size]
                clear_background_patch = clear_image_background[i:i + image_size, j:j + image_size]
                clear_patch = np.concatenate((clear_foreground_patch, clear_background_patch), axis=-1)


                x[sample_id] = blur_patch
                y[sample_id] = clear_patch
    

                sample_id += 1
                if sample_id == batch_size:
                    return x, y

class ValGenerator(Sequence):
    def __init__(self, image_dir):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        blur_image_dir = image_dir + '/' + 'blur'
        clear_image_dir = image_dir + '/' + 'clear'
        blur_image_paths  = [p for p in Path(blur_image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        clear_image_paths = [p for p in Path(clear_image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(blur_image_paths)
        self.data = []

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for blur_image_path in blur_image_paths:
            x = cv2.imread(str(blur_image_path))
            clear_image_path = str(blur_image_path).replace('blur', 'clear')
            y = cv2.imread(clear_image_path)

            # add clear_mask
            ret, y_mask =  cv2.threshold(y, 127, 1, cv2.THRESH_BINARY)
            y_fg = y * (1 - y_mask)
            y_bg = y * y_mask


            h, w, _ = x.shape
            x = x[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)

            # modify: concatenate foreground and background
            y_fg = y_fg[:(h // 16) * 16, :(w // 16) * 16]
            y_bg = y_bg[:(h // 16) * 16, :(w // 16) * 16]
            y_cont = np.concatenate((y_fg, y_bg), axis=-1)

            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y_cont, axis=0)])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
