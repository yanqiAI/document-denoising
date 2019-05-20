This repository contains the full source codes which are used to denoise the document dirty image.

## Pre-requisites

- python 3.6
- Tensorflow
- keras 
- OpenCV

## dataset prepare
1. prepare clear iamges and some background iamges
2. run /script/create_blue_data.py  this images have big and different sizes
3. run /script/crop_dataset.py  get train dataset default size 256 * 256

## train model
python train.py 

In model.py has 3 different models: srresnet, srresnet+, unet, default model is srresnet, srresnet+ has some problem.

## test model

python test_model.py  save blur_image | foreground_image | background_image | denoising_image, test image size is 256 * 256
![Image text](https://github.com/yanqiAI/document-denoising/blob/master/img/test_model/0141747.jpg)

python test.py save denoising_image, test image size is unconstrained.