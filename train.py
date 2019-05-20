import argparse
import numpy as np
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from model import get_model, PSNR, PSNR_FG_BG, SSIM, L0Loss, L1_loss, Total_loss, UpdateAnnealingParameter
from generator import NoisyImageGenerator, ValGenerator, DeblurImageGenerator
from noise_model import get_noise_model
from keras.callbacks import TensorBoard
from IPython.core.debugger import Tracer

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--blur_image_dir", type=str, default='E:/datasets/data515/blur',
                        help="train image dir")
    parser.add_argument('--clear_image_dir', type=str, default='E:/datasets/data515/clear',
                        help='clear image dir')
    parser.add_argument("--val_dir", type=str, default='E:/datasets/data515',
                        help="validation image dir")
    parser.add_argument("--image_size", type=int, default=64,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=10,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mae",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    # parser.add_argument("--source_noise_model", type=str, default="clean",
    #                     help="noise model for source images")
    # parser.add_argument("--target_noise_model", type=str, default="clean",
    #                     help="noise model for target images")
    # parser.add_argument("--val_noise_model", type=str, default="clean",
    #                     help="noise model for validation source images")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'srresnet+' or 'unet')")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    blur_image_dir = args.blur_image_dir
    clear_image_dir = args.clear_image_dir
    val_dir = args.val_dir
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    loss_type = args.loss
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    model = get_model(args.model) # srresnet  output

    if args.weight is not None:
        model.load_weights(args.weight)

    opt = Adam(lr=lr)
    callbacks = []

    if loss_type == "l0":
        l0 = L0Loss()
        callbacks.append(UpdateAnnealingParameter(l0.gamma, nb_epochs, verbose=1))
        loss_type = l0()

    generator = DeblurImageGenerator(blur_image_dir, clear_image_dir, batch_size=batch_size,
                                    image_size=image_size)

    # model outputs====>denoising results
    #clear_outputs = model.outputs

    # inputs labels
    #train_image_patch = generator.flow_from_directory(...)

    #model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])
    model.compile(optimizer=opt,
                  loss=L1_loss,
                  metrics=[PSNR, SSIM])

    # source_noise_model = get_noise_model(args.source_noise_model)
    # target_noise_model = get_noise_model(args.target_noise_model)
    # val_noise_model = get_noise_model(args.val_noise_model)

    # generator = NoisyImageGenerator(blur_image_dir, source_noise_model, target_noise_model, batch_size=batch_size,
    #                                 image_size=image_size)

    generator = DeblurImageGenerator(blur_image_dir, clear_image_dir, batch_size=batch_size,
                                    image_size=image_size)

    val_generator = ValGenerator(val_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_PSNR:.5f}.hdf5",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
                                     save_best_only=True))

    callbacks.append(TensorBoard(log_dir='./logs',
                                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                 write_graph=True,  # 是否存储网络结构图
                                 write_grads=True,  # 是否可视化梯度直方图
                                 write_images=True,  # 是否可视化参数
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None)
                     )

    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
