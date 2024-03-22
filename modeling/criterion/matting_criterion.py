import cv2
import torch
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]

class MattingCriterion(nn.Module):
    def __init__(self,
                 *,
                 losses,
                 ):
        super(MattingCriterion, self).__init__()
        self.losses = losses

    def loss_gradient_penalty(self, sample_map ,preds, targets):
        preds = preds['phas']
        targets = targets['phas']

        #sample_map for unknown area
        # scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x, delta_gt_x) + \
            F.l1_loss(delta_pred_y, delta_gt_y) + \
            0.01 * torch.mean(torch.abs(delta_pred_x)) +  \
            0.01 * torch.mean(torch.abs(delta_pred_y)))

        return dict(loss_gradient_penalty=loss)

    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):

        scale = sample_map.shape[0]*262144/torch.sum(sample_map)
        # scale = 1

        loss = F.l1_loss(preds['phas']*sample_map, targets['phas']*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1

        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)

    def l1_loss(self ,sample_map, preds, targets):
        preds = preds['phas']
        targets = targets['phas']

        # scale = sample_map.shape[0]*262144/torch.sum(sample_map)
        weight = get_unknown_tensor_from_pred(preds, 30, train_mode=True)

        loss = (regression_loss(preds, targets, loss_type='l1', weight=weight))

        return dict(l1_loss=loss)

    def forward(self, sample_map, preds, targets):
        losses = dict()
        for k in self.losses:
            if k=='l1_loss' or k=='loss_gradient_penalty':
                losses.update(getattr(self, k)(sample_map, preds, targets))
            else:
                losses.update(getattr(self, k)(preds, targets))
        return losses




#-----------l1 loss----------#
def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W
    N, C, H, W = pred.shape
    pred = F.interpolate(pred, size=(640,640), mode='nearest')
    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred < 1.0/255.0] = 0
    uncertain_area[pred > 1-1.0/255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1

    weight = np.array(weight, dtype=float)
    weight = torch.from_numpy(weight).cuda()

    weight = F.interpolate(weight, size=(H,W), mode='nearest')

    return weight

def regression_loss(logit, target, loss_type='l1', weight=None):
    """
    Alpha reconstruction loss
    :param logit:
    :param target:
    :param loss_type: "l1" or "l2"
    :param weight: tensor with shape [N,1,H,W] weights for each pixel
    :return:
    """
    if weight is None:
        if loss_type == 'l1':
            return F.l1_loss(logit, target)
        elif loss_type == 'l2':
            return F.mse_loss(logit, target)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
    else:
        if loss_type == 'l1':
            return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        elif loss_type == 'l2':
            return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))


#-----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]