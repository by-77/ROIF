import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict
from PIL import Image
from options import *
from model.ROIF import ROIF
from average_meter import AverageMeter


def train(model: ROIF,
          device: torch.device,
          net_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    train_data, val_data = utils.get_data_loaders(net_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], net_config.message_length))).to(device)
            losses, _ = model.train_on_batch([image, message])

            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        for image, _ in val_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], net_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_tensors) = model.validate_on_batch([image, message])
            for name, loss in losses.items():
                validation_losses[name].update(loss)
            if first_iteration:
                if net_config.enable_fp16:
                    # 将输入图像和中间结果转换为 float 以进行半精度计算
                    image = image.half()
                    encoded_images = encoded_images.half()
                    noised_images = noised_images.half()

                    # 计算原始图像和编码图像之间的差异
                    image_difference = (image - encoded_images).abs()
                    print('原始图像差异:',image_difference)


                    utils.save_images(
                        #original_images=image[:images_to_save, :, :, :].cpu(),
                        watermarked_images=encoded_images[:images_to_save, :, :, :].cpu(),
                        #noised_images=noised_images[:images_to_save, :, :, :].cpu(),
                        epoch=epoch,
                        folder=os.path.join(this_run_folder, 'images'),
                        resize_to=(saved_images_size if net_config.enable_fp16 else None)  # 确保只传递一次
                    )

                first_iteration = False


        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
