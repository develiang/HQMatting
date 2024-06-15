import os
from PIL import Image
from tqdm import tqdm
import cv2

alpha_path = "/data/dataset/FaceImage/test_mtcnn/V1/inpaint_mat_skin_transfer_AdaIN_alpha"
image_path =  "/data/dataset/FaceImage/test_mtcnn/V1/inpaint_mat_skin_transfer_AdaIN"
res_path = "/data/dataset/FaceImage/test_mtcnn/V1/inpaint_mat_skin_transfer_AdaIN_bg"



def change_bg(image, alpha, bg_color):
    # image = Image.fromarray(image, mode='RGB')

    # image_ = Image.open("/data/jlguo/file/qwz_2024.3.19_test/test_10000_SR_O_BG/cropped_img_0042_enhance.png")
    # alpha = Image.open(alpha_path)
    # size = tuple(image.size)
    # size = (image.shape[1], image.shape[0])
    new_image = Image.new("RGBA", image.size, bg_color)
    # new_image = Image.new("RGBA", image.size, (0,0,0,0))

    new_image.paste(image, (0, 0), alpha)

    # new_image.save(save_path, "PNG")

    return new_image

for name in tqdm(os.listdir(image_path)):
    image = Image.open(os.path.join(image_path, name))
    alpha = Image.open(os.path.join(alpha_path, name))

    res = change_bg(image,alpha, (40,190,249,255))

    res.save(os.path.join(res_path, name))

