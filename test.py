import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import cv2
import sys
sys.path.append("/data/jlguo/Code/segment-anything/")
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from os.path import join as opj
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser
from segment_anything import sam_model_registry, SamPredictor
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import warnings
warnings.filterwarnings('ignore')


def init_model(model, checkpoint, device):
    # sourcery skip: extract-duplicate-method, extract-method, inline-immediately-returned-variable
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    assert model in ['vitmatte-s', 'vitmatte-b']
    if model == 'vitmatte-s':
        config = 'configs/common/model.py'
        cfg = LazyConfig.load(config)
        model = instantiate(cfg.model)
        model.to('cuda')
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    elif model == 'vitmatte-b':
        config = 'configs/common/model.py'
        cfg = LazyConfig.load(config)
        cfg.model.backbone.embed_dim = 768
        cfg.model.backbone.num_heads = 12
        cfg.model.decoder.in_chans = 768
        model = instantiate(cfg.model)
        model.to('cuda')
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    return model

def get_data(image, mask):
    """
    Get the data of one image.
    Input:
        image_dir: the directory of the image
        trimap_dir: the directory of the trimap
    """
    # image = Image.open(image_dir).convert('RGB')
    image = F.to_tensor(image).unsqueeze(0)
    # mask = Image.open(mask_dir).convert('L')
    mask = F.to_tensor(mask).unsqueeze(0)
    # i = mask.max()

    return {
        'image': image,
        'mask': mask
    }

def infer_one_image(model, input):
    """
    Infer the alpha matte of one image.
    Input:
        model: the trained model
        image: the input image
        trimap: the input trimap
    """
    output = model(input)['phas'].flatten(0, 2)
    output = F.to_pil_image(output)
    # output.save(opj(save_dir))

    return output


def change_bg(image, alpha, bg_color):
    image = Image.fromarray(image, mode='RGB')

    # image_ = Image.open("/data/jlguo/file/qwz_2024.3.19_test/test_10000_SR_O_BG/cropped_img_0042_enhance.png")
    # alpha = Image.open(alpha_path)
    # size = tuple(image.size)
    # size = (image.shape[1], image.shape[0])
    new_image = Image.new("RGBA", image.size, bg_color)
    # new_image = Image.new("RGBA", image.size, (0,0,0,0))

    new_image.paste(image, (0, 0), alpha)

    # new_image.save(save_path, "PNG")

    return new_image


def test(input):
    return

def img_sr(args,input):
    img = input["input"]
    if args.face_enhance:
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        output, _ = upsampler.enhance(img, outscale=args.outscale)

    return output


if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    #-------------test path-------------#
    parser.add_argument('--test_path', default="/data/jlguo/file/qwz_2024.3.19_test/test_100000",type=str, required=False)
    parser.add_argument('--save_path', default="/data/jlguo/file/qwz_2024.3.19_test/test_10000_SR_with_esrgan",type=str, required=False)

    #---------------use model init-----------------#
    parser.add_argument('--use_sr', default=True,type=bool, required=False)

    parser.add_argument('--matting_type', default="vitmatte-s",type=str, required=False)
    parser.add_argument('--matting_checkpoint', default="/data/jlguo/Code/HQ-matting/pretrained/model_final.pth",type=str, required=False)

    parser.add_argument('--sr_type', default="RealESRGAN_x4plus",type=str, required=False)
    parser.add_argument('--outscale', default=4, type=int, required=False)
    parser.add_argument('--sr_checkpoint', default="/data/jlguo/Code/HQ-matting/pretrained/RealESRGAN_x4plus.pth",type=str, required=False)

    parser.add_argument('--sam_checkpoint', default="/data/jlguo/Code/segment-anything/pretrained/sam_vit_h_4b8939.pth",type=str, required=False)
    parser.add_argument('--model_type', default="vit_h",type=str, required=False)

    parser.add_argument('--bg_color', default=(40,190,249,255),type=int, required=False)

    #--------------others-----------------------------#
    parser.add_argument('--device', default="cuda",type=str, required=False)
    #---------------sr args--------------------#
    parser.add_argument('--face_enhance', type=bool, default=False, help='Use GFPGAN to enhance face')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    args = parser.parse_args()
    # matting_inference(
    #     config_dir = args.config_dir,
    #     checkpoint_dir = args.checkpoint_dir,
    #     inference_dir = args.inference_dir,
    #     data_name = args.data_name,
    #     data_dir = args.data_dir
    # )

    if args.use_sr:
        sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=args.outscale,
            model_path=args.sr_checkpoint,
            model=sr_model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=not args.fp32)
        if args.face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
        #init sr model
        # pred sr img


    #init sam and matting model
    input_box = np.array([0, 30, 480, 640])
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    mattting_model = init_model(args.matting_type, args.matting_checkpoint, args.device)

    # test
    for name in tqdm(os.listdir(args.test_path)):
        image = cv2.imread(os.path.join(args.test_path, name))
        # img_sr after sr result (origin size)
        if args.use_sr:
            sr_input = {"input":image}
            image_sr = img_sr(args, sr_input)
            image_sr = cv2.resize(image_sr, (480,640), interpolation=None)

        # predict mask
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
        )
        mask = np.float32(masks[0])

        #matting
        if args.use_sr:
            input = get_data(image_sr, mask)
            alpha = infer_one_image(mattting_model, input)
            img_change_bg = change_bg(image ,alpha, args.bg_color)
            img_change_bg.save(os.path.join(args.save_path, name), "PNG")





        # if args.use_sr:
        #     input = {}


