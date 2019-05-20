# coding:utf-8
from keras.models import Model
from keras.layers import Input, Add, PReLU, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K
from keras import losses
from keras.applications.vgg16 import VGG16
import numpy as np
import tensorflow as tf
from IPython.core.debugger import Tracer


class L0Loss:
    def __init__(self):
        self.gamma = K.variable(2.)

    def __call__(self):
        def calc_loss(y_true, y_pred):
            loss = K.pow(K.abs(y_true - y_pred) + 1e-8, self.gamma)
            return loss
        return calc_loss

class UpdateAnnealingParameter(Callback):
    def __init__(self, gamma, nb_epochs, verbose=0):
        super(UpdateAnnealingParameter, self).__init__()
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        K.set_value(self.gamma, new_gamma)

        if self.verbose > 0:
            print('\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s.' % (epoch + 1, new_gamma))

'''
add vgg loss
'''

def preproces_vgg(x):
    # scale from [-1, 1] to [0, 255]
    x += 1.
    x *= 127.5

    # RGB -> BGR
    x = x[..., ::-1]
    # apply Imagenet preprocessing : BGR mean
    mean = [103.939, 116.778, 123.68]
    _IMAGENET_MEAN = K.constant(-np.array(mean))
    x = K.bias_add(x, K.cast(_IMAGENET_MEAN, K.dtype(x)))
    return x

def VGG_loss(y_true, y_pred):
    # load pretrained VGG
    vgg16 = VGG16(include_top=False,
                  input_shape=(64, 64, 3),
                  weights='imagenet')

    vgg16.trainable = False
    for l in vgg16.layers:
        l.trainable = False

    # create a model that output the features from level 'block2_conv2'
    features_extractor = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block2_conv2').output)

    features_true = features_extractor(preproces_vgg(y_true))
    features_pred = features_extractor(preproces_vgg(y_pred))

    return K.mean(K.square(features_pred - features_true), axis=[1, 2, 3])

def Total_loss(y_true, y_pred):
    '''
    TODO: not work when callbacks now!!! Because the two loss shape are different (2, 64, 64) vs (2, 32, 32)
    modify axis=[-1]====>axis=[1,2,3]  loss shape (2,) vs (2,) vector
    '''
    l1_loss = K.mean(K.abs(y_pred - y_true), axis=[1, 2, 3])
    vgg_loss = VGG_loss(y_true, y_pred)
    total_loss = l1_loss + 0.006 * vgg_loss

    return total_loss

'''
add foreground and background l1 loss
'''
def L1_loss(y_true, y_pred):
    y_fg_true = y_true[:, :, :, :3]
    y_bg_true = y_true[:, :, :, 3:]

    y_fg_pred = y_pred[:, :, :, :3]
    y_bg_pred = y_pred[:, :, :, 3:]

    l1_fg_loss = losses.mean_absolute_error(y_fg_true, y_fg_pred)
    l1_bg_loss = losses.mean_absolute_error(y_bg_true, y_bg_pred)

    total_loss = 0.8 * l1_fg_loss + 0.2 * l1_bg_loss

    return total_loss

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 255.0)

def PSNR_FG_BG(y_true, y_pred):
    y_fg_true = y_true[:, :, :, :3]
    y_bg_true = y_true[:, :, :, 3:]

    y_fg_pred = y_pred[:, :, :, :3]
    y_bg_pred = y_pred[:, :, :, 3:]

    PSNR_FG = PSNR(y_fg_true, y_fg_pred)
    PSNR_BG = PSNR(y_bg_true, y_bg_pred)

    return (PSNR_FG + PSNR_BG) / 2.


def get_model(model_name="srresnet"):
    if model_name == "srresnet":
        return get_srresnet_model()
    elif model_name =="srresnet+":
        return get_srresnet_model_plus()
    elif model_name == "unet":
        return get_unet_model(out_ch=3)
    else:
        raise ValueError("model_name should be 'srresnet'or 'unet'")

# SRResNet
def get_srresnet_model(input_channel_num=3, feature_dim=64, resunit_num=16):
    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])

        return m

    inputs = Input(shape=(None, None, input_channel_num))


    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])
    x = Conv2D(input_channel_num * 2, (3, 3), padding="same", kernel_initializer="he_normal")(x)


    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model

# SRResNet+
'''
First, SRResNet+ additionally takes a noise level map M as input. 
Second, SRResNet+ increases the number of feature maps from 64 to 96. 
Third, SRResNet+ removes the batch normalization layer
'''
def get_srresnet_model_plus(input_channel_num=3, feature_dim=96, resunit_num=16):
    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        #x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        #x = BatchNormalization()(x)
        m = Add()([x, inputs])

        return m

    inputs = Input(shape=(None, None, input_channel_num))
    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)

    x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    #x = BatchNormalization()(x)
    x = Add()([x, x0])
    x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)

    return model

# UNet: code from https://github.com/pietz/unet-keras
def get_unet_model(input_channel_num=3, out_ch=3, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    def _conv_block(m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, activation=acti, padding='same')(m)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, activation=acti, padding='same')(n)
        n = BatchNormalization()(n) if bn else n

        return Concatenate()([m, n]) if res else n

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = _conv_block(m, dim, acti, bn, res)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same')(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            n = Concatenate()([n, m])
            m = _conv_block(n, dim, acti, bn, res)
        else:
            m = _conv_block(m, dim, acti, bn, res, do)

        return m

    i = Input(shape=(None, None, input_channel_num))
    o = _level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1)(o)
    model = Model(inputs=i, outputs=o)

    return model

def main():
    model = get_model("unet")
    model.summary()

if __name__ == '__main__':
    main()
