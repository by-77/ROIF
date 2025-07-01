import torch.nn
import argparse
import os
import math
import utils
from model.ROIF import *
from PIL import Image
import torchvision.transforms.functional as TF
from Noise_Layer import ScreenShooting
from argparse import Namespace
def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def yuv_psnr(img):
    imgy = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2:, :, :]
    imgu = -0.14713 * img[:, 0, :, :] + (-0.28886) * img[:, 1, :, :] + 0.436 * img[:, 2:, :, :]
    imgv = 0.615 * img[:, 0, :, :] + -0.51499 * img[:, 1, :, :] + (-0.10001) * img[:, 2:, :, :]
    return imgy, imgu, imgv


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source_images', '-s', required=True, type=str,
                        help='The image to watermark')
    parser.add_argument("--noise", '-n', nargs="*", )

    args = parser.parse_args()
    source_images_dir = args.source_images


    source_image = 'ILSVRC2012_val_00003014.JPEG'


    full_path = os.path.join(source_images_dir, source_image)


    try:
        image_pil = Image.open(full_path)

    except PermissionError as e:
        print(f"Permission denied for file: {full_path}")
        print(e)
    except FileNotFoundError as e:
        print(f"File not found: {full_path}")
        print(e)
    finally:

        if 'image_pil' in locals() and image_pil:
            image_pil.close()

    train_options, net_config, noise_config= utils.load_options(args.options_file)

    noiser = ScreenShooting()

    checkpoint = torch.load(args.checkpoint_file)
    hidden_net = ROIF(net_config, device,  None)
    utils.model_from_checkpoint(hidden_net, checkpoint)
    source_images = os.listdir(args.source_images)

    for source_image in source_images:
        image_pil = Image.open(os.path.join(args.source_images, source_image))#image_pil = Image.open(args.source_images + source_image)
        image_pil = image_pil.resize((net_config.H, net_config.W))
        image_tensor = TF.to_tensor(image_pil).to(device)
        image_tensor = image_tensor * 2 - 1
        image_tensor.unsqueeze_(0)
        np.random.seed(42)
        message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                         net_config.message_length))).to(device)

        losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch(
            [image_tensor, message])

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        message_detached = message.detach().cpu().numpy()
        print('original: {}'.format(message_detached))
        print('decoded : {}'.format(decoded_rounded))
        print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))

    utils.save_images(image_tensor.cpu(), encoded_images.cpu(), noised_images.cpu(),'test', 'C:/Users/28/Desktop/haoARWGAN-main/测试图像', resize_to=(128, 128))


if __name__ == '__main__':
    main()
