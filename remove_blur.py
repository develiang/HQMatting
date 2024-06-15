import os
os.environ["CUDA_VISIBLE_DEVICES"]='7'
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
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import warnings
warnings.filterwarnings('ignore')


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
    parser.add_argument('--test_path', default="/data/jlguo/Code/HQ-matting/my_pic/others/ysl.png",type=str, required=False)
    parser.add_argument('--save_sr_path', default="/data/jlguo/Code/HQ-matting/my_pic/others/ysl-sr.png",type=str, required=False)
    # parser.add_argument('--save_alpha_path', default="/data/jlguo/Code/HQ-matting/test/vitmatte/alpha",type=str, required=False)

    parser.add_argument('--save_alpha_path', default="/data/jlguo/Code/HQ-matting/test/vitmatte_trimap/alpha",type=str, required=False)

    parser.add_argument('--save_alpha', default=0,type=bool, required=False)
    parser.add_argument('--save_mask', default=1,type=bool, required=False)
    parser.add_argument('--save_change_bg', default=0,type=bool, required=False)


    #---------------use model init-----------------#
    parser.add_argument('--use_sr', default=True,type=bool, required=False)

    parser.add_argument('--sr_type', default="RealESRGAN_x4plus",type=str, required=False)
    parser.add_argument('--outscale', default=4, type=int, required=False)
    parser.add_argument('--sr_checkpoint', default="/data/jlguo/Code/HQ-matting/pretrained/RealESRGAN_x4plus.pth",type=str, required=False)

    parser.add_argument('--bg_color', default=(40,190,249),type=int, required=False)
    # parser.add_argument('--bg_color', default=(0,0,0,255),type=int, required=False)

    #--------------others-----------------------------#
    parser.add_argument('--device', default="cuda",type=str, required=False)
    #---------------sr args--------------------#
    parser.add_argument('--face_enhance', type=bool, default=True, help='Use GFPGAN to enhance face')
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

    # os.makedirs(args.save_change_bg_path, exist_ok=True)
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

    # test
    if os.path.isfile(args.test_path):
        image = cv2.imread(args.test_path)
        H,W,_ = image.shape
        image= cv2.resize(image, (W//4,H//4), interpolation=None)


        sr_input = {"input":image}
        image_sr = img_sr(args, sr_input)
        # image_sr = cv2.resize(image_sr, (W,H), interpolation=None)

        cv2.imwrite(args.save_sr_path, image_sr)

    else:
        for name in tqdm(os.listdir(args.test_path)):
            image = cv2.imread(os.path.join(args.test_path, name))
            H,W,_ = image.shape
            # img_sr after sr result (origin size)
            sr_input = {"input":image}
            image_sr = img_sr(args, sr_input)
            image_sr = cv2.resize(image_sr, (W,H), interpolation=None)

            cv2.imwrite(os.path.join(args.save_sr_path, name), image_sr)

            # predict mask
            # image_sr.save(os.path.join(args.save_change_bg_path, name), "PNG")





        # if args.use_sr:
        #     input = {}


