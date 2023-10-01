# %%
import cv2
import numpy as np
import pathlib
import torch

from models.network_swinir import SwinIR as net


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

private_model_zoo = {
    'classical_2x_M': {
        'file_name': '001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth',
        'model_type': 'classical_sr',
        'scale': 2,
        'large_model': False,
    },

    'classical_2x_light': {
        'file_name': '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
        'model_type': 'lightweight_sr',
        'scale': 2,
        'large_model': False,
    },

    'real_2x_light': {
        'file_name': '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_PSNR.pth',
        'model_type': 'real_sr',
        'scale': 2,
        'large_model': False,
    },
}


private_image_zoo = {
    't1': {
        'file_name': 'F0D9843E-5F6C-47FA-8172-F74E3423F321_1_105_c.jpeg'
    }
}


def get_private_model_path(model_name):
    model_file_name = private_model_zoo[model_name]['file_name']
    private_model_zoo_path = pathlib.Path(
        '/Users/marlinberger/Desktop/local/Coding/SwinIR/model_zoo'
    )
    model_path = private_model_zoo_path / model_file_name
    return model_path


def get_model(model_name):
    model_type = private_model_zoo[model_name]['model_type']
    upsale = private_model_zoo[model_name]['scale']
    large_model = private_model_zoo[model_name]['large_model']

    # look up in description in main_test_swinir; seems not to be important
    # parameter
    default_classical_sr_training_patch_size = 128

    if model_type == 'classical_sr':
        model = net(upscale=upsale, in_chans=3,
                    img_size=default_classical_sr_training_patch_size,
                    window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                    upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif model_type == 'lightweight_sr':
        model = net(upscale=upsale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60,
                    num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect',
                    resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif model_type == 'real_sr':
        if not large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=upsale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6],
                        embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv',
                        resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory;
            # use ema for GAN training
            model = net(upscale=upsale, in_chans=3, img_size=64,
                        window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                        embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv',
                        resi_connection='3conv')
        param_key_g = 'params_ema'

    pretrained_model = torch.load(
        get_private_model_path(model_name)
    )

    model.load_state_dict(
        pretrained_model[param_key_g] if param_key_g in pretrained_model.keys()
        else pretrained_model, strict=True)

    model.eval()
    model = model.to(DEVICE)

    return model


def preprocess_following_original_git(img_lq):
    """create pseudo-lq image: just load the image we want to infer, but
    process it according the authors git to enable easy inference
    """
    standard_window_size = 8  # this only differs for jpeg-comp-art-reduction
    window_size = standard_window_size

    img_lq = img_lq / 255.

    # HCW-BGR to CHW-RGB
    img_lq = np.transpose(
        img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))

    # CHW-RGB to NCHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(DEVICE)

    # pad input image to be a multiple of window_size
    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_lq = torch.cat(
        [img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    img_lq = torch.cat(
        [img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

    return img_lq


def get_image(image_name):
    image_file_name = private_image_zoo[image_name]['file_name']
    private_image_zoo_path = pathlib.Path(
        '/Users/marlinberger/Desktop/local/Coding/SwinIR/image_zoo'
    )
    image_path = private_image_zoo_path / image_file_name
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR).astype(np.float32)
    return preprocess_following_original_git(img)


def predict(model_name, image_name):
    model = get_model(model_name)
    img = get_image(image_name)

    print(img.shape)

    with torch.no_grad():
        output = model(img)

    print(output.shape)



if __name__ == '__main__':
    predict(
        'classical_2x_light',
        't1'
    )
# %%